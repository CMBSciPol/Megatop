import argparse
import tracemalloc

# from mpi4py.futures import MPICommExecutor
from pathlib import Path

import healpy as hp
import megabuster as mb
import numpy as np
import pymaster as nmt
from mpi4py import MPI

from megatop import Config, DataManager
from megatop.utils import Timer, logger, mask
from megatop.utils.binning import load_nmt_binning
from megatop.utils.mpi import MPISUM, get_world
from megatop.utils.preproc import common_beam_and_nside
from megatop.utils.spectra import (
    compute_auto_cross_cl_from_maps_list,
    get_common_beam_wpix,
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
    logger.warning("Normalizing analysis mask to 1, TODO: remove after merge")
    # TODO: remove after merge
    mask_analysis /= np.max(mask_analysis)  # normalize the mask to 1
    binary_mask = hp.read_map(manager.path_to_binary_mask).astype(bool)

    # Loading component separation operator
    if not config.parametric_sep_pars.use_megabuster:
        W_maxL = np.load(manager.get_path_to_compsep_results(sub=id_sim_sky), allow_pickle=True)[
            "W_maxL"
        ]

    # Loading bin info from map2cl step:
    nmt_bins = load_nmt_binning(manager)

    # Getting effective beam TODO: add case for input maps (no preproc)
    effective_beam_CMB = get_common_beam_wpix(
        config.pre_proc_pars.common_beam_correction, config.nside
    )
    # effective_beam_CMB = np.ones_like(effective_beam_CMB)  # No beam for now

    logger.warning(
        "We are only using the CMB effective beam in the noise spectra estimation\nIf you want to use the effective beam for the other components, please update the code"
    )

    if config.parametric_sep_pars.use_megabuster:
        with Timer("init-megabuster"):
            logger.warning(
                "Using Megabuster for component separation, make sure to have the correct parameters set in the config file"
            )

            obsmat_operator_fname = manager.get_path_list_or_None("suffix_obsmat_scipy")

            if np.all(np.array(obsmat_operator_fname) == Path()):
                # TODO: how to handle this case?
                logger.warning(
                    "No observation matrix file provided. Using identity matrix instead."
                )
                obsmat_operator_rhs = None
            elif np.any(np.array(obsmat_operator_fname) == Path()):
                msg_any_obs = "Not all observation matrix files are provided. Provide either all or none, partial set of observation matrices is not supported. A temporary solution is to provide a identity observation matrix for channels without filtering"
                raise ValueError(msg_any_obs)
            else:
                logger.debug(f"Loading observation matrix from {obsmat_operator_fname}")
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

            path_eigen_decomp_fname = manager.get_path_list_or_None("suffix_eigen_decomp")
            if np.all(np.array(path_eigen_decomp_fname) == Path()):
                # TODO: how to handle this case?
                logger.warning(
                    "No observation matrix file provided for CG. Using identity matrix instead."
                )
                central_freq_op = None
                matrix_precond = None
            elif np.any(np.array(path_eigen_decomp_fname) == Path()):
                msg_any_obs = "Not all eigen decomposition files are provided. Provide either all or none, partial set of eigen decomposition for each frequency is not supported. A temporary solution is to provide a identity observation matrix for channels without filtering"
                raise ValueError(msg_any_obs)
            else:
                logger.debug(f"Loading observation matrix from {obsmat_operator_fname}")
                central_freq_op = mb.tools.get_dense_furax_operator_from_freq_array(
                    mb.io.load_matrix_precond(path_eigen_decomp_fname, power_diagonal=1)
                )
                matrix_precond = mb.io.load_matrix_precond(
                    path_eigen_decomp_fname, power_diagonal=-1
                )

            with Timer("load-covmat"):
                noisecov_fname = manager.path_to_pixel_noisecov
                logger.debug(f"Loading covmat from {noisecov_fname}")
                noisecov = np.load(noisecov_fname)

            noisecov_QU_masked = mask.apply_binary_mask(noisecov[:, 1:], binary_mask, unseen=False)
            inverse_noisecov_QU_masked = np.zeros_like(noisecov_QU_masked)
            inverse_noisecov_QU_masked[noisecov_QU_masked != 0] = (
                1.0 / noisecov_QU_masked[noisecov_QU_masked != 0]
            )

            # get the 'options' through the appropriate method which returns a dict
            parameters_foregrounds_x = np.load(
                manager.get_path_to_compsep_results(sub=id_sim_sky), allow_pickle=True
            )["x"]
            parameters_foregrounds = {
                "beta_dust": np.array(parameters_foregrounds_x[0]),
                "beta_pl": np.array(parameters_foregrounds_x[1]),
            }

    # Initializing workspace
    with Timer("init-namaster-workspace"):
        fields_init_wsp = nmt.NmtField(
            mask_analysis,
            None,
            spin=2,
            beam=effective_beam_CMB[:-1],
            purify_e=config.map2cl_pars.purify_e,
            purify_b=config.map2cl_pars.purify_b,
            n_iter=config.map2cl_pars.n_iter_namaster,
        )
        workspaceff = nmt.NmtWorkspace.from_fields(fields_init_wsp, fields_init_wsp, nmt_bins)

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
        # for comp in range(W_maxL.shape[0]):
        for freq in range(W_maxL.shape[1]):
            dict_comp_WmaxL_freq = {"CMB": W_maxL[0, freq, :], "Dust": W_maxL[1, freq, :]}
            if config.parametric_sep_pars.include_synchrotron:
                dict_comp_WmaxL_freq["Synch"] = W_maxL[2, freq, :]
            all_Cls_WmaxL_freq = compute_auto_cross_cl_from_maps_list(
                dict_comp_WmaxL_freq,
                mask_analysis,
                effective_beam_CMB[:-1],
                workspaceff,
                purify_e=config.map2cl_pars.purify_e,
                purify_b=config.map2cl_pars.purify_b,
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
        # import IPython; IPython.embed()
        # Applying component-separation operator
        if not config.parametric_sep_pars.use_megabuster:
            noise_map_post_compsep = np.einsum(
                "ifsp,fsp->isp", W_maxL, noise_freq_maps_preprocessed[:, 1:]
            )  # slicing noise to remove T
        else:
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
                central_freq_op=central_freq_op,
                matrix_precond=matrix_precond,
                dictionary_parameters_CG={
                    "max_steps_CG": megabuster_options["max_steps_CG"],
                    "tol_CG": megabuster_options["tol_CG"],
                },
                ordering_parameter=["beta_dust", "beta_pl"],
                ordering_component=["cmb", "dust", "synchrotron"],
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
        noise_Cls = compute_auto_cross_cl_from_maps_list(
            noise_comp_dict,
            mask_analysis,
            effective_beam_CMB[:-1],
            workspaceff,
            purify_e=config.map2cl_pars.purify_e,
            purify_b=config.map2cl_pars.purify_b,
            n_iter=config.map2cl_pars.n_iter_namaster,
            inverse_effective_transfer_function=inverse_normalized_Cl_effective_TF,
            # inverse_effective_transfer_function=inverse_effective_transfer_function,
        )
        # import IPython; IPython.embed()
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
