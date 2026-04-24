import argparse
import tracemalloc
from pathlib import Path

import healpy as hp
import numpy as np

from megatop import Config, DataManager
from megatop.utils import Timer, logger
from megatop.utils.binning import load_nmt_binning
from megatop.utils.mpi import MPISUM, get_world
from megatop.utils.spectra import (
    compute_auto_cross_cl_from_maps_dict,
    get_common_beam_wpix,
    initialize_nmt_workspace,
    limit_namaster_output,
)
from megatop.utils.utils import MemoryUsage


def init_workspace(config: Config, manager: DataManager):
    analysis_mask = hp.read_map(manager.path_to_analysis_mask)
    nmt_bins = load_nmt_binning(manager)

    # Getting effective beam TODO: add case for input maps (no preproc)
    effective_beam_CMB = get_common_beam_wpix(
        config.pre_proc_pars.common_beam_correction, config.nside, config.lmax
    )
    logger.warning(
        "We are only using the CMB effective beam in the noise spectra estimation\nIf you want to use the effective beam for the other components, please update the code"
    )
    # Initializing workspace
    with Timer("init-namaster-workspace"):
        workspace = initialize_nmt_workspace(
            nmt_bins=nmt_bins,
            analysis_mask=analysis_mask,
            beam=effective_beam_CMB,
            purify_e=config.map2cl_pars.purify_e,
            purify_b=config.map2cl_pars.purify_b,
            n_iter=config.map2cl_pars.n_iter_namaster,
            lmax=config.lmax,
        )
    return workspace


def noise_spectra_estimator(
    config: Config, manager: DataManager, workspace_nmt, id_sim_sky: int | None = None
):
    tracemalloc.start()

    comm, rank, size = get_world()
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
    analysis_mask = hp.read_map(manager.path_to_analysis_mask)
    binary_mask = hp.read_map(manager.path_to_binary_mask).astype(bool)

    # Loading component separation operator
    W_maxL = np.load(manager.get_path_to_compsep_results(id_sim_sky), allow_pickle=True)["W_maxL"]

    # Loading bin info from map2cl step:
    nmt_bins = load_nmt_binning(manager)

    if (
        config.pre_proc_pars.correct_for_TF and config.parametric_sep_pars.use_harmonic_compsep
    ) and not config.parametric_sep_pars.alm2map:
        logger.info("Computing effective Transfer Function after component separation")
        transfer_freq = []
        for tf_path in manager.get_TF_filenames():
            transfer = np.load(tf_path, allow_pickle=True)["full_tf"]
            transfer_freq.append(transfer)
        transfer_freq = np.array(transfer_freq)

        Cl_WmaxL = np.zeros(
            (W_maxL.shape[0], W_maxL.shape[0], W_maxL.shape[1], 4, nmt_bins.get_n_bands())
        )
        for freq in range(W_maxL.shape[1]):
            dict_comp_WmaxL_freq = {"CMB": W_maxL[0, freq, :], "Dust": W_maxL[1, freq, :]}
            if config.parametric_sep_pars.include_synchrotron:
                dict_comp_WmaxL_freq["Synch"] = W_maxL[2, freq, :]
            all_Cls_WmaxL_freq = compute_auto_cross_cl_from_maps_dict(
                maps_dict=dict_comp_WmaxL_freq,
                analysis_mask=analysis_mask,
                workspace=workspace_nmt,
            )
            Cl_WmaxL[0, 0, freq] = all_Cls_WmaxL_freq["CMBxCMB"]
            Cl_WmaxL[0, 1, freq] = all_Cls_WmaxL_freq["CMBxDust"]
            Cl_WmaxL[1, 0, freq] = all_Cls_WmaxL_freq["CMBxDust"]
            Cl_WmaxL[1, 1, freq] = all_Cls_WmaxL_freq["DustxDust"]
            if config.parametric_sep_pars.include_synchrotron:
                Cl_WmaxL[0, 2, freq] = all_Cls_WmaxL_freq["CMBxSynch"]
                Cl_WmaxL[2, 0, freq] = all_Cls_WmaxL_freq["CMBxSynch"]
                Cl_WmaxL[1, 2, freq] = all_Cls_WmaxL_freq["DustxSynch"]
                Cl_WmaxL[2, 1, freq] = all_Cls_WmaxL_freq["DustxSynch"]
                Cl_WmaxL[2, 2, freq] = all_Cls_WmaxL_freq["SynchxSynch"]

        Cl_effective_TF = np.einsum(
            "ckfsl, fspl, lpfkc-> ckspl", Cl_WmaxL, transfer_freq[:, -4:, -4:], Cl_WmaxL.T
        )  # keeping only polarised components
        normalisation_WCl = np.einsum("ckfsl, lpfkc-> ckspl", Cl_WmaxL, Cl_WmaxL.T)
        normalized_Cl_effective_TF = Cl_effective_TF / normalisation_WCl
        inverse_normalized_Cl_effective_TF = np.zeros_like(normalized_Cl_effective_TF)
        for i in range(normalized_Cl_effective_TF.shape[0]):
            for j in range(normalized_Cl_effective_TF.shape[1]):
                for ell in range(normalized_Cl_effective_TF.shape[-1]):
                    # Inverting the transfer function for each ell over the spectra dimension
                    inverse_normalized_Cl_effective_TF[i, j, :, :, ell] = np.linalg.inv(
                        normalized_Cl_effective_TF[i, j, :, :, ell]
                    )

        # effective_transfer_function, inverse_effective_transfer_function = (
        #     get_effective_transfer_function(transfer_freq, W_maxL, binary_mask))
    else:
        # inverse_effective_transfer_function = None
        inverse_normalized_Cl_effective_TF = None

    sum_noise_spectra = {}

    for id_realisation in rank_realisation_list:
        id_real = None if n_sim_noise is None else id_realisation

        logger.info(f"id_realisation = {id_real}")
        logger.info(f"in = {rank_realisation_list}")

        # TODO if use input maps for compsep then can also just import input noise maps here
        logger.info("Loading pre-processed noise maps")

        noise_freq_maps_preprocessed = np.load(manager.get_path_to_preprocessed_noise_maps(id_real))

        # Applying component-separation operator
        noise_map_post_compsep = np.einsum(
            "ifsp,fsp->isp",
            W_maxL,
            noise_freq_maps_preprocessed[:, 1:],
        )  # slicing noise to remove T #TODO: Any speed improvement ?
        noise_map_post_compsep *= binary_mask

        # TODO: update keys wrt relevant components once implemented in compsep step
        noise_comp_dict = {
            "Noise_CMB": noise_map_post_compsep[0],
            "Noise_Dust": noise_map_post_compsep[1],
        }
        if config.parametric_sep_pars.include_synchrotron:
            noise_comp_dict["Noise_Synch"] = noise_map_post_compsep[2]

        # Computing auto and cross spectra
        noise_Cls = compute_auto_cross_cl_from_maps_dict(
            maps_dict=noise_comp_dict,
            analysis_mask=analysis_mask,
            workspace=workspace_nmt,
            inverse_effective_transfer_function=inverse_normalized_Cl_effective_TF,
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
        bin_index_lminlmax = np.load(manager.path_to_binning, allow_pickle=True)[
            "bin_index_lminlmax"
        ]

        # Average noise spectra over nsims
        mean_noise_spectra = {}
        for key in sum_noise_spectra:
            mean_noise_spectra[key] = sum_noise_spectra_recvbuf[key] / int_n_sim_noise

        mean_noise_spectra = limit_namaster_output(mean_noise_spectra, bin_index_lminlmax)

    else:
        mean_noise_spectra = None

    if rank == root:
        fname = manager.get_path_to_noise_spectra_cross_components(id_sim_sky)
        logger.info(f"Saving estimated noise spectra to {fname}")
        np.savez(fname, **mean_noise_spectra)

    return id_sim_sky


def main():
    parser = argparse.ArgumentParser(description="Noise spectra estimator")
    parser.add_argument("--config", type=Path, required=True, help="config file")
    parser.add_argument("--sim", type=int, default=None, help="process only this simulation index")

    args = parser.parse_args()
    config = Config.load_yaml(args.config)
    manager = DataManager(config)

    world, rank, size = get_world()
    workspace_nmt = init_workspace(
        config, manager
    )  # Initialize the workspace once and for all (WARNING WHEN WE'LL NEED EFFECTIVE BEAMS !!!!!)
    if rank == 0:
        manager.dump_config()
        manager.create_output_dirs(config.map_sim_pars.n_sim, config.noise_sim_pars.n_sim)

    if args.sim is not None:
        noise_spectra_estimator(config, manager, workspace_nmt, id_sim_sky=args.sim)
        return

    n_sim_sky = config.map_sim_pars.n_sim
    if n_sim_sky == 0:
        noise_spectra_estimator(config, manager, workspace_nmt)
    else:
        for i in range(n_sim_sky):
            result = noise_spectra_estimator(config, manager, workspace_nmt, i)
            logger.info(
                f"Finished noise spectra estimation for sky simulation {result + 1}/{n_sim_sky}"
            )


if __name__ == "__main__":
    main()
