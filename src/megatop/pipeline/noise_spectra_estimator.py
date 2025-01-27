import argparse
import os
import tracemalloc

import healpy as hp
import numpy as np
import pymaster as nmt
from mpi4py import MPI

from megatop.utils.metadata_manager import BBmeta
from megatop.utils.mpi import MPISUM
from megatop.utils.preproc import common_beam_and_nside
from megatop.utils.spectra import (
    compute_auto_cross_cl_from_maps_list,
    get_common_beam_wpix,
    initialize_nmt_workspace,
)
from megatop.utils.utils import MemoryUsage


def noise_spectra_estimator(meta):
    tracemalloc.start()

    try:
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.rank
        root = 0

    except (ModuleNotFoundError, ImportError) as e:
        # Error handling
        meta.logger.info(f"ERROR IN MPI:{e}")
        meta.logger.info("Proceeding without MPI\n")

        root = 0
        rank = 0
        size = 1

    MemoryUsage(meta, f"rank = {rank} ")

    nreal = meta.noise_cov_pars["nrealisation_noise_cov"]

    # The None case of nreal is useful when calling get_noise_map_filename
    # so we need to handle it when creating the list of realisations to loop over
    int_nreal = 1 if nreal is None else nreal
    realisation_list = np.arange(int_nreal)

    # splitting the list of simulation between the ranks of the process:
    rank_realisation_list = np.array_split(realisation_list, size)[rank]

    # Loading masks
    mask_analysis = meta.read_mask("analysis")
    binary_mask = meta.read_mask("binary").astype(bool)

    # Loading component separation operator
    W_maxL = np.load(
        os.path.join(meta.components_directory, "comp_sep_results.npz"), allow_pickle=True
    )["W_maxL"]

    # Loading bin info from map2cl step:
    binning_info = np.load(os.path.join(meta.spectra_directory, "binning.npz"), allow_pickle=True)
    nmt_bins = nmt.NmtBin.from_edges(binning_info["bin_low"], binning_info["bin_high"] + 1)

    # Getting effective beam TODO: add case for input maps (no preproc)
    effective_beam = get_common_beam_wpix(meta.pre_proc_pars["common_beam_correction"], meta.nside)

    # Initializing workspace
    meta.timer.start("initializing_workspace")
    path_Cl_lens = meta.get_fname_cls_fiducial_cmb("lensed")

    workspaceff = initialize_nmt_workspace(
        nmt_bins,
        path_Cl_lens,
        meta.nside,
        mask_analysis,
        effective_beam[:-1],
        meta.map2cl_pars["purify_e"],
        meta.map2cl_pars["purify_b"],
        meta.map2cl_pars["n_iter_namaster"],
    )
    meta.timer.stop("initializing_workspace", "Initializing workspace")

    sum_noise_spectra = None

    for id_realisation in rank_realisation_list:
        noise_freq_maps = []

        id_real = None if nreal is None else id_realisation

        meta.logger.info(f"id_realisation = {id_real}")
        meta.logger.info(f"in = {rank_realisation_list}")

        if meta.noise_cov_pars["save_preprocessed_noise_maps"]:
            # TODO if use input maps for compsep then can also just import input noise maps here
            meta.logger.info("Loading pre-processed noise maps")

            add_sim_id = f"_{id_real:04d}.npy" if nreal is not None else ".npy"
            noise_freq_maps_preprocessed = np.load(
                os.path.join(meta.covmat_directory, "freq_noise_maps_preprocessed" + add_sim_id)
            )

        else:
            maps_list = meta.maps_list
            nside_in_list = []
            for m in maps_list:
                path_noise_map = meta.get_noise_map_filename(m, id_sim=id_real)

                meta.logger.debug(f"Importing noise map: {path_noise_map}")

                noise_freq_maps.append(hp.read_map(path_noise_map, field=None).tolist())
                nside_in_list.append(hp.get_nside(noise_freq_maps[-1][-1]))

            if np.all(
                np.array(meta.pre_proc_pars["common_beam_correction"])
                == np.array(meta.beams_FWHM_arcmin)
            ):
                meta.logger.info(
                    "Common beam correction is the same as the input beam, no need to apply it."
                )
                meta.logger.info(
                    "WARNING: this is mostly for testing it might not actually represent the real noise"
                )

                noise_freq_maps_preprocessed = noise_freq_maps

            else:
                noise_freq_maps = np.array(noise_freq_maps, dtype=object)
                noise_freq_maps_preprocessed = common_beam_and_nside(meta, noise_freq_maps)

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
            effective_beam,
            workspaceff,
            purify_e=meta.map2cl_pars["purify_e"],
            purify_b=meta.map2cl_pars["purify_b"],
            n_iter=meta.map2cl_pars["n_iter_namaster"],
        )

        # Initializing the mean noise spectra dict if doesn't exist
        if not sum_noise_spectra:
            sum_noise_spectra = {}
            for key in noise_Cls:
                sum_noise_spectra[key] = np.zeros_like(noise_Cls[key])

        # Summing the noise spectra
        for key in noise_Cls:
            sum_noise_spectra[key] += noise_Cls[key]

    sum_noise_spectra_recvbuf = {}
    for key in sum_noise_spectra:
        sum_noise_spectra_recvbuf[key] = MPISUM(sum_noise_spectra[key], comm, rank, root)

    if rank == root:
        # Average noise spectra over nsims
        mean_noise_spectra = {}
        for key in sum_noise_spectra:
            mean_noise_spectra[key] = sum_noise_spectra_recvbuf[key] / int_nreal

    else:
        mean_noise_spectra = None

    if rank == root:
        np.savez(
            os.path.join(meta.spectra_directory, "Noise_cross_components_Cls.npz"),
            **mean_noise_spectra,
        )

    if rank == root:
        meta.logger.info("\n\nNoise spectra computation step completed successfully.\n\n")

    return 0


def main():
    parser = argparse.ArgumentParser(description="simplistic simulator")  # TODO change name ?
    parser.add_argument("--globals", type=str, help="Path to yaml with global parameters")

    args = parser.parse_args()
    meta = BBmeta(args.globals)
    noise_spectra_estimator(meta)


if __name__ == "__main__":
    main()
