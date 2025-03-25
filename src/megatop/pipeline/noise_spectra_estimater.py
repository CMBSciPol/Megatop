import argparse
import tracemalloc

# from mpi4py.futures import MPICommExecutor
from pathlib import Path

import healpy as hp
import numpy as np
import pymaster as nmt
from mpi4py import MPI

from megatop import Config, DataManager
from megatop.utils import Timer, logger
from megatop.utils.mpi import MPISUM, get_world
from megatop.utils.preproc import common_beam_and_nside
from megatop.utils.spectra import (
    compute_auto_cross_cl_from_maps_list,
    get_common_beam_wpix,
    initialize_nmt_workspace,
    limit_namaster_output,
)
from megatop.utils.utils import MemoryUsage


def noise_spectra_estimator(config: Config, manager: DataManager, id_sim_sky: int | None = None):
    tracemalloc.start()

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.rank
    root = 0

    MemoryUsage(f"rank = {rank} ")

    n_sim_noise = config.noise_sim_pars.n_sim

    # The None case of nreal is useful when calling get_innoise_map_filename
    # so we need to handle it when creating the list of realisations to loop over
    int_n_sim_noise = 1 if n_sim_noise is None else n_sim_noise
    realisation_list = np.arange(int_n_sim_noise)

    # splitting the list of simulation between the ranks of the process:
    rank_realisation_list = np.array_split(realisation_list, size)[rank]

    # Loading masks
    mask_analysis = hp.read_map(manager.path_to_analysis_mask)
    binary_mask = hp.read_map(manager.path_to_binary_mask).astype(bool)

    # Loading component separation operator
    W_maxL = np.load(manager.get_path_to_compsep_results(sub=id_sim_sky), allow_pickle=True)[
        "W_maxL"
    ]

    # Loading bin info from map2cl step:
    binning_info = np.load(manager.get_path_to_spectra_binning(sub=id_sim_sky), allow_pickle=True)
    nmt_bins = nmt.NmtBin.from_edges(binning_info["bin_low"], binning_info["bin_high"] + 1)

    # Getting effective beam TODO: add case for input maps (no preproc)
    effective_beam_CMB = get_common_beam_wpix(
        config.pre_proc_pars.common_beam_correction, config.nside
    )

    logger.warning(
        "We are only using the CMB effective beam in the noise spectra estimation\nIf you want to use the effective beam for the other components, please update the code"
    )
    # Initializing workspace
    with Timer("init-namaster-workspace"):
        workspaceff = initialize_nmt_workspace(
            nmt_bins,
            manager.path_to_lensed_scalar,
            config.nside,
            mask_analysis,
            effective_beam_CMB[:-1],
            config.map2cl_pars.purify_e,
            config.map2cl_pars.purify_b,
            config.map2cl_pars.n_iter_namaster,
        )

    sum_noise_spectra = {}

    for id_realisation in rank_realisation_list:
        noise_freq_maps = []

        id_real = None if n_sim_noise is None else id_realisation

        logger.info(f"id_realisation = {id_real}")
        logger.info(f"in = {rank_realisation_list}")

        if config.noise_cov_pars.save_preprocessed_noise_maps:
            # TODO if use input maps for compsep then can also just import input noise maps here
            logger.info("Loading pre-processed noise maps")

            noise_freq_maps_preprocessed = np.load(
                manager.get_path_to_preprocessed_noise_maps(id_real)
            )

        else:
            nside_in_list = []
            for noise_filename in manager.get_noise_maps_filenames(id_real):
                logger.debug(f"Importing noise map: {noise_filename}")
                noise_freq_maps.append(hp.read_map(noise_filename, field=None).tolist())
                nside_in_list.append(hp.get_nside(noise_freq_maps[-1][-1]))

            if np.all(
                np.array(config.pre_proc_pars.common_beam_correction) == np.array(config.beams)
            ):
                logger.info(
                    "Common beam correction is the same as the input beam, no need to apply it."
                )
                logger.info(
                    "WARNING: this is mostly for testing it might not actually represent the real noise"
                )

                noise_freq_maps_preprocessed = noise_freq_maps

            else:
                noise_freq_maps = np.array(noise_freq_maps, dtype=object)
                noise_freq_maps_preprocessed = common_beam_and_nside(
                    nside=config.nside,
                    common_beam=config.pre_proc_pars.common_beam_correction,
                    frequency_beams=config.beams,
                    freq_maps=noise_freq_maps,
                )

        # Applying component-separation operator
        noise_map_post_compsep = np.einsum(
            "ifsp,fsp->isp", W_maxL, noise_freq_maps_preprocessed[:, 1:]
        )  # slicing noise to remove T
        noise_map_post_compsep *= binary_mask

        # TODO: update keys wrt relevant components once implemented in compsep step
        noise_comp_dict = {
            "Noise_CMB": noise_map_post_compsep[0],
            "Noise_Dust": noise_map_post_compsep[1],
            "Noise_Synch": noise_map_post_compsep[2],
        }

        # Computing auto and cross spectra
        noise_Cls = compute_auto_cross_cl_from_maps_list(
            noise_comp_dict,
            mask_analysis,
            effective_beam_CMB[:-1],
            workspaceff,
            purify_e=config.map2cl_pars.purify_e,
            purify_b=config.map2cl_pars.purify_b,
            n_iter=config.map2cl_pars.n_iter_namaster,
        )

        # Summing the noise spectra
        for key in noise_Cls:
            if key not in sum_noise_spectra:
                sum_noise_spectra[key] = np.zeros_like(noise_Cls[key])
            sum_noise_spectra[key] += noise_Cls[key]

    # Perform the reduction
    if comm is not None:
        sum_noise_spectra_recvbuf = {
            k: MPISUM(val, comm, rank, root) for k, val in sum_noise_spectra.items()
        }
    else:
        sum_noise_spectra_recvbuf = sum_noise_spectra

    if rank == root:
        # Average noise spectra over nsims
        mean_noise_spectra = {}
        for key in sum_noise_spectra:
            mean_noise_spectra[key] = sum_noise_spectra_recvbuf[key] / int_n_sim_noise
        mean_noise_spectra = limit_namaster_output(
            mean_noise_spectra, binning_info["bin_index_lminlmax"]
        )

    else:
        mean_noise_spectra = None

    if rank == root:
        path = manager.get_path_to_noise_spectra(sub=id_sim_sky)
        path.mkdir(parents=True, exist_ok=True)
        fname = manager.get_path_to_noise_spectra_cross_components(sub=id_sim_sky)
        logger.info(f"Saving estimated noise spectra to {fname}")
        np.savez(fname, **mean_noise_spectra)

    return id_sim_sky


def main():
    parser = argparse.ArgumentParser(description="Noise spectra estimator")
    parser.add_argument("--config", type=Path, required=True, help="config file")

    args = parser.parse_args()
    config = Config.load_yaml(args.config)
    manager = DataManager(config)

    world, rank, size = get_world()
    if rank == 0:
        manager.dump_config()

    n_sim_sky = config.map_sim_pars.n_sim
    if n_sim_sky == 0:
        noise_spectra_estimator(config, manager)
    else:
        for i in range(n_sim_sky):
            result = noise_spectra_estimator(config, manager, i)
            logger.info(
                f"Finished noise spectra estimation for sky simulation {result + 1}/{n_sim_sky}"
            )


if __name__ == "__main__":
    main()
