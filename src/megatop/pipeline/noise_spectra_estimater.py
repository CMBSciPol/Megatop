import argparse
import os
import tracemalloc
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import healpy as hp  # noqa: E402
import megabuster as mb  # noqa: E402
import numpy as np  # noqa: E402
from fgbuster.separation_recipes import _format_alms, _r_to_c_alms  # noqa: E402

from megatop import Config, DataManager  # noqa: E402
from megatop.pipeline.map_to_cler import get_inv_binned_MCM  # noqa: E402
from megatop.utils import Timer, logger  # noqa: E402
from megatop.utils.binning import load_nmt_binning  # noqa: E402
from megatop.utils.mask import apply_binary_mask  # noqa: E402
from megatop.utils.mpi import MPISUM, get_world  # noqa: E402
from megatop.utils.spectra import (  # noqa: E402
    compute_auto_cross_cl_from_alms_list_MCMverstion,
    compute_auto_cross_cl_from_maps_dict,  # noqa: E402
    get_common_beam_wpix,  # noqa: E402
    initialize_nmt_workspace,  # noqa: E402
    limit_namaster_output,  # noqa: E402
)  # noqa: E402
from megatop.utils.utils import MemoryUsage  # noqa: E402


def debug_cutscales(noise_freq_maps_preprocessed, nside):
    # TODO: TEST
    logger.warning("TEST: Applying smooth cut at large scales to noise component maps")
    logger.warning("TEST: DOING IT AFTER PARAM ESTIMATION (can't do before for noise maps)")

    def get_smooth_scale_cut(cut_scale, smoothing_scale, lmax, lmin=0):
        ell = np.arange(lmax + 1)
        smooth_cut = 0.5 * (1 + np.tanh((ell - cut_scale) / smoothing_scale))
        smooth_cut[:lmin] = 0.0
        return smooth_cut

    cut_array = get_smooth_scale_cut(30, 1, lmax=3 * nside)
    freq_maps_cut = np.zeros_like(noise_freq_maps_preprocessed)
    for f in range(noise_freq_maps_preprocessed.shape[0]):
        alm_comp = hp.map2alm(
            [
                noise_freq_maps_preprocessed[f, 0],
                noise_freq_maps_preprocessed[f, 1],
                noise_freq_maps_preprocessed[f, 2],
            ],
            lmax=3 * nside,
        )
        for s in range(alm_comp.shape[0]):
            hp.almxfl(alm_comp[s], cut_array, inplace=True)
        freq_maps_cut[f] = hp.alm2map(
            alm_comp, nside=nside, lmax=3 * nside, pol=True
        )  # removing temperature

    return freq_maps_cut


def initialize_megabuster(manager, binary_mask, id_sim_sky):
    with Timer("init-megabuster"):
        logger.warning(
            "Using Megabuster for component separation, make sure to have the correct parameters set in the config file"
        )

        try:
            parameters_foregrounds_x = np.load(
                manager.get_path_to_compsep_results(id_sim_sky), allow_pickle=True
            )["x"]
        except FileNotFoundError:
            logger.error(
                f"Results from comp sep not found for {manager.get_path_to_compsep_results(id_sim_sky)}"
            )
            return id_sim_sky

        obsmat_operator_fname = manager.get_path_list_or_None("suffix_obsmat_scipy")

        if np.all(np.array(obsmat_operator_fname) == Path()):
            # TODO: how to handle this case?
            logger.warning("No observation matrix file provided. Using identity matrix instead.")
            obsmat_operator_rhs = None
        elif np.any(np.array(obsmat_operator_fname) == Path()):
            msg_any_obs = "Not all observation matrix files are provided. Provide either all or none, partial set of observation matrices is not supported. A temporary solution is to provide a identity observation matrix for channels without filtering"
            raise ValueError(msg_any_obs)
        else:
            logger.info(f"Loading observation matrix from {obsmat_operator_fname}")
            npix = binary_mask.size
            indices_mask = np.arange(npix)[hp.reorder(binary_mask, r2n=True) != 0]
            mask_stacked_nest = np.hstack((indices_mask + npix, indices_mask + 2 * npix))
            obsmat_operator_rhs = mb.io.build_obsmat_operator_from_flattened_matrices(
                mb.io.load_all_obsmat(
                    obsmat_operator_fname,
                    size_obsmat=3 * npix,
                    kind="precomputations_scipy",
                    mask_stacked=mask_stacked_nest,
                ),
                nstokes=2,
                return_transpose=True,
            )

        indices_freq_not_filtered = manager.get_indices_frequencies_not_filtered()

        with Timer("load-covmat"):
            noisecov_fname = manager.path_to_pixel_noisecov
            logger.debug(f"Loading covmat from {noisecov_fname}")
            noisecov = np.load(noisecov_fname)

        noisecov_QU_masked = apply_binary_mask(noisecov[:, 1:], binary_mask, unseen=False)
        inverse_noisecov_QU_masked = np.zeros_like(noisecov_QU_masked)
        inverse_noisecov_QU_masked[noisecov_QU_masked != 0] = (
            1.0 / noisecov_QU_masked[noisecov_QU_masked != 0]
        )
        

        path_eigen_decomp_fname = manager.get_path_list_or_None("suffix_eigen_decomp")
        if np.all(np.array(path_eigen_decomp_fname) == Path()):
            # TODO: how to handle this case?
            logger.warning(
                "No observation matrix file provided for CG. Using identity matrix instead."
            )
            matrix_central = None
            matrix_precond = None
            common_filtered_eigenvector = None
        # elif np.any(np.array(path_eigen_decomp_fname) == Path()):
        #     msg_any_obs = "Not all eigen decomposition files are provided. Provide either all or none, partial set of eigen decomposition for each frequency is not supported. A temporary solution is to provide a identity observation matrix for channels without filtering"
        #     raise ValueError(msg_any_obs)
        else:
            logger.debug(f"Loading observation matrix from {obsmat_operator_fname}")

            n_pix_masked = binary_mask[binary_mask!=0].size

            inverse_noisecov_QU_masked_nested = np.zeros_like(inverse_noisecov_QU_masked)
            
            n_freq = np.array(config.frequencies).size
            
            for i in range(n_freq):
                inverse_noisecov_QU_masked_nested[i,...] = hp.reorder(inverse_noisecov_QU_masked[i], r2n=True)

            inverse_noisecov_QU_masked_nested = np.array(inverse_noisecov_QU_masked_nested)

            matrix_central = mb.io.load_matrix_precond(
                path_eigen_decomp_fname, 
                power_diagonal=1,
                diagonal_array_fill=inverse_noisecov_QU_masked_nested,
                n_pix_diagonal=n_pix_masked
            )

            matrix_precond = mb.io.load_matrix_precond(
                path_eigen_decomp_fname, 
                power_diagonal=-1,
                diagonal_array_fill=inverse_noisecov_QU_masked_nested,
                n_pix_diagonal=n_pix_masked
            )

            if indices_freq_not_filtered is not None:
                common_basis_reference_freq = [f for f in range(n_freq) if f not in indices_freq_not_filtered][0]
            else:
                common_basis_reference_freq = 0

            add_ = ''
            if not str(path_matrix_eigen_decomp[idx_freq]).endswith('.npz'):
                add_ = '.npz'
            
            common_filtered_eigenvector = np.load(path_eigen_decomp_fname[common_basis_reference_freq]+add_)['eigvecs']

        

        # get the 'options' through the appropriate method which returns a dict

        parameters_foregrounds = {
            "beta_dust": np.array(parameters_foregrounds_x[0]),
            "beta_pl": np.array(parameters_foregrounds_x[1]),
        }

        
    return (
        parameters_foregrounds,
        obsmat_operator_rhs,
        matrix_central,
        matrix_precond,
        inverse_noisecov_QU_masked,
        common_filtered_eigenvector,
        indices_freq_not_filtered
    )


def get_effective_transfer_function(
    config, manager, W_maxL, nmt_bins, effective_beam_CMB, analysis_mask, workspace_nmt
):
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
            beam=effective_beam_CMB,
            n_iter=config.map2cl_pars.n_iter_namaster,
            lmax=config.lmax,
            purify_b=config.map2cl_pars.purify_b,
            purify_e=config.map2cl_pars.purify_e,
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
    return normalized_Cl_effective_TF, inverse_normalized_Cl_effective_TF


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
    return workspace, effective_beam_CMB


def noise_spectra_estimator(
    config: Config,
    manager: DataManager,
    workspace_nmt,
    effective_beam_CMB,
    id_sim_sky: int | None = None,
):
    # ========================== Sky simulation/observation initializations =========================
    # We first start by initializing the parameters that are sky simulation/observation dependent before looping over noise realisations.
    # For instance the weighing matrix after compsep that will depend on data.

    tracemalloc.start()

    comm, rank, size = get_world()
    root = 0

    MemoryUsage(f"rank = {rank} ")

    n_sim_noise = config.noise_sim_pars.n_sim

    # The None case of nreal is useful when calling get_innoise_map_filename
    # so we need to handle it when creating the list of realisations to loop over
    int_n_sim_noise = (
        1
        if n_sim_noise is None or config.noise_sim_pars.DEBUG_save_TRUEnoise_simulations
        else n_sim_noise
    )
    realisation_list = np.arange(int_n_sim_noise)

    # splitting the list of simulation between the ranks of the process:
    rank_noise_realisation_list = np.array_split(realisation_list, size)[rank]

    # Loading masks
    analysis_mask = hp.read_map(manager.path_to_analysis_mask)
    binary_mask = hp.read_map(manager.path_to_binary_mask).astype(bool)

    # Loading component separation operator
    if not config.parametric_sep_pars.use_megabuster:
        W_maxL = np.load(manager.get_path_to_compsep_results(id_sim_sky), allow_pickle=True)[
            "W_maxL"
        ]
    else:
        # TODO: TEST
        logger.info(f"Hostname: {os.uname().nodename}")
        logger.info(f"JAX version: {jax.__version__}")
        logger.info(f"JAX backend: {jax.default_backend()}")
        logger.info(f"JAX devices: {jax.devices()}")
        logger.info(f"CPU affinity: {len(os.sched_getaffinity(0))}")
        logger.info(f"OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS')}")
        logger.info(f"OPENBLAS_NUM_THREADS: {os.environ.get('OPENBLAS_NUM_THREADS')}")
        (
            parameters_foregrounds,
            obsmat_operator_rhs,
            matrix_central,
            matrix_precond,
            inverse_noisecov_QU_masked,
            common_filtered_eigenvector,
            indices_freq_not_filtered,
        ) = initialize_megabuster(manager, binary_mask, id_sim_sky)

    MemoryUsage(f"rank = {rank} ")

    use_alm = (
        config.map2cl_pars.use_harmonic_output_alm
        and config.parametric_sep_pars.use_harmonic_compsep
        and config.pre_proc_pars.DEBUGHARMONIC_preproc_noise_TF_correction_alm
    )

    # if use_alm:
    #     W_maxL_lm = np.load(manager.get_path_to_compsep_results(id_sim_sky), allow_pickle=True)[
    #         "W_maxL_lm"
    #     ]

    # Loading bin info from map2cl step:
    nmt_bins = load_nmt_binning(manager)
    inv_MCM = get_inv_binned_MCM(workspace_nmt, nmt_bins)

    MemoryUsage(f"rank = {rank} ")

    if (
        config.pre_proc_pars.correct_for_TF and config.parametric_sep_pars.use_harmonic_compsep
    ) and not config.parametric_sep_pars.alm2map:
        logger.info("Computing effective transfer function after compsep")
        _, inverse_normalized_Cl_effective_TF = get_effective_transfer_function(
            config, manager, W_maxL, nmt_bins, effective_beam_CMB, analysis_mask, workspace_nmt
        )

    else:
        inverse_normalized_Cl_effective_TF = None

    sum_noise_spectra = {}
    MemoryUsage(f"rank = {rank} ")

    # ========================= Loop over noise realisations =========================
    # Now that the sky simulation/observation dependent parameters are initialized, we can loop over the noise realisations to compute the noise spectra after component separation.

    for id_realisation in rank_noise_realisation_list:
        MemoryUsage(f"rank = {rank} ")

        id_real = None if n_sim_noise is None else id_realisation

        logger.info(f"id_realisation = {id_real}")
        logger.info(f"in = {rank_noise_realisation_list}")

        # if config.noise_cov_pars.save_preprocessed_noise_maps:
        logger.info("Loading pre-processed noise maps")
        if config.noise_sim_pars.DEBUG_save_TRUEnoise_simulations:
            logger.info("Loading TRUE noise simulations saved during preproc step")
            path_true_preproc_noise_maps = manager.get_path_to_preprocessed_TRUE_noise_maps(
                id_sim_sky
            )
            logger.info(f"Loading TRUE noise simulations from {path_true_preproc_noise_maps}")
            noise_freq_maps_preprocessed = np.load(path_true_preproc_noise_maps)

        else:
            noise_freq_maps_preprocessed = np.load(
                manager.get_path_to_preprocessed_noise_maps(id_real)
            )
        noise_freq_maps_preprocessed_ = noise_freq_maps_preprocessed.copy()  # I have to do this otherwise the ruff check isn't happy at the debug_cutscales step, it says noise_freq_maps_preprocessed is undefined...

        # Applying component-separation operator
        if not use_alm:
            logger.info(
                "Using preprocessed noise maps to compute noise spectra after component separation"
            )

            if not config.parametric_sep_pars.use_megabuster:
                noise_map_post_compsep = np.einsum(
                    "ifsp,fsp->isp",
                    W_maxL,
                    noise_freq_maps_preprocessed[:, 1:],
                )  # slicing noise to remove T #TODO: Any speed improvement ?
            else:
                if config.map2cl_pars.DEBUG_cut_scales:
                    noise_freq_alms_preprocessed = debug_cutscales(
                        noise_freq_maps_preprocessed_, config.nside
                    )

                megabuster_options = config.parametric_sep_pars.get_megabuster_options_as_dict()
                noise_map_post_compsep = mb.compsep.perform_compsep(
                    first_guess_params=parameters_foregrounds,
                    fixed_params={"temp_dust": 20.0},
                    sky_map=noise_freq_maps_preprocessed[:, 1:] * binary_mask,
                    frequencies=np.array(config.frequencies),
                    invN_matrix=inverse_noisecov_QU_masked,
                    do_minimization=False,
                    binary_mask=binary_mask,
                    obs_mat_operator=None,
                    obsmat_operator_rhs=obsmat_operator_rhs,
                    use_preconditioner_diag=megabuster_options["use_preconditioner_diag"],
                    use_preconditioner_pinv=megabuster_options["use_preconditioner_pinv"],
                    use_preconditioner_qr=megabuster_options["use_preconditioner_qr"],
                    use_preconditioner_common_basis=megabuster_options["use_preconditioner_common_basis"],
                    matrix_central=matrix_central,
                    matrix_precond=matrix_precond,
                    common_filtered_eigenvector=common_filtered_eigenvector,
                    dictionary_parameters_CG={
                        "max_steps_CG": megabuster_options["max_steps_CG"],
                        "tol_CG": megabuster_options["tol_CG"],
                    },
                    ordering_parameter=["beta_dust", "beta_pl"],
                    ordering_component=["cmb", "dust", "synchrotron"],
                    indices_freq_not_filtered=indices_freq_not_filtered,
                ).s

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
                beam=effective_beam_CMB,
                n_iter=config.map2cl_pars.n_iter_namaster,
                lmax=config.lmax,
                purify_b=config.map2cl_pars.purify_b,
                purify_e=config.map2cl_pars.purify_e,
                inverse_effective_transfer_function=inverse_normalized_Cl_effective_TF,
            )
        else:
            logger.warning("THIS WHOLE OPTION IS NOT WELL TESTED AND NOT UP TO DATE!!!")
            logger.info(
                "Using preprocessed noise alms to compute noise spectra after component separation"
            )
            noise_freq_alms_preprocessed = np.load(
                manager.get_path_to_preprocessed_noise_alms(id_real)
            )
            W_maxL_lm_real_cross = np.load(
                manager.get_path_to_compsep_results(id_sim_sky), allow_pickle=True
            )["W_maxL_lm_real"]
            alm_postcompsep_real_cross = np.einsum(
                # "lebcf, lbf -> lec", W_maxL_lm_real_cross, _format_alms(data_alms_lmin_save.copy())
                "lebcf, lbf -> lec",
                W_maxL_lm_real_cross,
                _format_alms(noise_freq_alms_preprocessed.copy()),
            )
            alm_postcompsep_cross = _r_to_c_alms(alm_postcompsep_real_cross.T)
            noise_comp_dict_alm = {
                "Noise_CMB": alm_postcompsep_cross[0],
                "Noise_Dust": alm_postcompsep_cross[1],
            }
            if config.parametric_sep_pars.include_synchrotron:
                noise_comp_dict_alm["Noise_Synch"] = alm_postcompsep_cross[2]

            noise_Cls = compute_auto_cross_cl_from_alms_list_MCMverstion(
                noise_comp_dict_alm,
                nmt_bins,
                inv_MCM,
                config.lmax,
                effective_beam_CMB,
                DEBUG_ignore_MCM=config.map2cl_pars.DEBUG_ignore_MCM,
            )

        # Summing the noise spectra
        for key in noise_Cls:
            if key not in sum_noise_spectra:
                sum_noise_spectra[key] = np.zeros_like(noise_Cls[key])
            sum_noise_spectra[key] += noise_Cls[key]
        MemoryUsage(f"rank = {rank} ")
        logger.info(f"Finished realisation {id_realisation} on rank {rank}\n")
    logger.info("Finished all realisations assigned to this rank")

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

        bin_index_lminlmax = np.arange(len(mean_noise_spectra[key][0]))  # keeping all bins
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
    workspace_nmt, effective_beam_CMB = init_workspace(
        config, manager
    )  # Initialize the workspace once and for all (WARNING WHEN WE'LL NEED EFFECTIVE BEAMS !!!!!)
    if rank == 0:
        manager.dump_config()
        manager.create_output_dirs(config.map_sim_pars.n_sim, config.noise_sim_pars.n_sim)

    if args.sim is not None:
        noise_spectra_estimator(
            config, manager, workspace_nmt, effective_beam_CMB, id_sim_sky=args.sim
        )
        return

    n_sim_sky = config.map_sim_pars.n_sim
    if n_sim_sky == 0:
        noise_spectra_estimator(config, manager, workspace_nmt, effective_beam_CMB)
    else:
        for i in range(n_sim_sky):
            result = noise_spectra_estimator(config, manager, workspace_nmt, effective_beam_CMB, i)
            logger.info(
                f"Finished noise spectra estimation for sky simulation {result + 1}/{n_sim_sky}"
            )


if __name__ == "__main__":
    main()
