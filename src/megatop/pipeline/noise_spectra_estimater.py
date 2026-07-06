import argparse
import tracemalloc
from pathlib import Path
import pickle

import healpy as hp
import numpy as np
import jax
import jax.numpy as jnp
import megabuster as mb  # noqa: E402

from megatop import Config, DataManager
from megatop.utils import Timer, logger, mask
from megatop.utils.binning import load_nmt_binning
from megatop.utils.mpi import MPISUM, get_world
from megatop.utils.spectra import (
    compute_auto_cross_cl_from_maps_dict,
    get_common_beam_wpix,
    initialize_nmt_workspace,
    limit_namaster_output,
)
from megatop.utils.utils import MemoryUsage
from furax.obs.stokes import Stokes


def init_workspace(config: Config, manager: DataManager):
    analysis_mask = hp.read_map(manager.path_to_analysis_mask)
    nmt_bins = load_nmt_binning(manager)

    effective_beam_CMB = get_common_beam_wpix(
        config.pre_proc_pars.common_beam_correction, config.nside, config.lmax
    )
    logger.warning(
        "We are only using the CMB effective beam in the noise spectra estimation\nIf you want to use the effective beam for the other components, please update the code"
    )

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
    tracemalloc.start()

    comm, rank, size = get_world()
    root = 0

    MemoryUsage(f"rank = {rank} ")

    n_sim_noise = config.noise_sim_pars.n_sim
    int_n_sim_noise = 1 if n_sim_noise is None else n_sim_noise
    realisation_list = np.arange(int_n_sim_noise)
    rank_realisation_list = np.array_split(realisation_list, size)[rank]

    analysis_mask = hp.read_map(manager.path_to_analysis_mask)
    binary_mask = hp.read_map(manager.path_to_binary_mask).astype(bool)

    if not config.parametric_sep_pars.use_megabuster:
        W_maxL = np.load(manager.get_path_to_compsep_results(sub=id_sim_sky), allow_pickle=True)[
            "W_maxL"
        ]

    nmt_bins = load_nmt_binning(manager)

    if config.parametric_sep_pars.use_megabuster:
        with Timer("init-megabuster"):
            logger.warning(
                "Using Megabuster for component separation, make sure to have the correct parameters set in the config file"
            )
            try:
                parameters_foregrounds_x = np.load(
                    manager.get_path_to_compsep_results(id_sim_sky), allow_pickle=True
                )["x"]
                params_names = np.load(
                    manager.get_path_to_compsep_results(id_sim_sky), allow_pickle=True
                )["params_names"]
                parameters_dict = {name: parameters_foregrounds_x[i] for i, name in enumerate(params_names)}
            except FileNotFoundError:
                logger.error(
                    f"Results from comp sep not found for {manager.get_path_to_compsep_results(id_sim_sky)}"
                )
                return id_sim_sky

            if config.parametric_sep_pars.megabuster_options.use_obsmat:
                obsmat_operator_fname = manager.get_path_list_or_None("suffix_obsmat_scipy")
                if np.any(np.array(obsmat_operator_fname) == Path()):
                    raise ValueError("Not all observation matrix files are provided.")
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
                path_eigen_decomp_fname = manager.get_path_list_or_None("suffix_eigen_decomp")
                if np.any(np.array(path_eigen_decomp_fname) == Path()):
                    raise ValueError("Not all eigen decomposition files are provided.")
                else:
                    logger.debug(f"Loading observation matrix from {obsmat_operator_fname}")
                    central_freq_op = mb.tools.get_dense_furax_operator_from_freq_array(
                        mb.io.load_matrix_precond(path_eigen_decomp_fname, power_diagonal=1)
                    )
                    matrix_precond = mb.io.load_matrix_precond(
                        path_eigen_decomp_fname, power_diagonal=-1
                    )
            else:
                central_freq_op = None
                matrix_precond = None
                obsmat_operator_rhs = None

            with Timer("load-covmat"):
                noisecov_fname = manager.path_to_pixel_noisecov
                logger.debug(f"Loading covmat from {noisecov_fname}")
                noisecov = np.load(noisecov_fname)

            noisecov_QU_masked = mask.apply_binary_mask(noisecov[:, 1:], binary_mask, unseen=False)
            inverse_noisecov_QU_masked = np.zeros_like(noisecov_QU_masked)
            inverse_noisecov_QU_masked[noisecov_QU_masked != 0] = (
                1.0 / noisecov_QU_masked[noisecov_QU_masked != 0]
            )

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
        )
        normalisation_WCl = np.einsum("ckfsl, lpfkc-> ckspl", Cl_WmaxL, Cl_WmaxL.T)
        normalized_Cl_effective_TF = Cl_effective_TF / normalisation_WCl
        inverse_normalized_Cl_effective_TF = np.zeros_like(normalized_Cl_effective_TF)
        for i in range(normalized_Cl_effective_TF.shape[0]):
            for j in range(normalized_Cl_effective_TF.shape[1]):
                for ell in range(normalized_Cl_effective_TF.shape[-1]):
                    inverse_normalized_Cl_effective_TF[i, j, :, :, ell] = np.linalg.inv(
                        normalized_Cl_effective_TF[i, j, :, :, ell]
                    )
    else:
        inverse_normalized_Cl_effective_TF = None

    sum_noise_spectra = {}

    # Charger l'opérateur une seule fois
    fname_operator = manager.get_path_to_compsep_results(id_sim_sky).with_suffix('.pkl')
    with open(fname_operator, "rb") as f:
        diagonal_central_term = pickle.load(f)

    # Charger toutes les cartes de bruit d'un coup
    all_noise_Q = []
    all_noise_U = []
    for id_realisation in rank_realisation_list:
        id_real = None if n_sim_noise is None else id_realisation
        if config.pre_proc_pars.use_real_beams:
            noise_freq_maps = np.load(manager.get_path_to_real_preprocessed_noise_maps(id_real))
        else:
            noise_freq_maps = np.load(manager.get_path_to_preprocessed_noise_maps(id_real))
        noise_QU = noise_freq_maps[:, 1:] * binary_mask
        all_noise_Q.append(noise_QU[:, 0, :])
        all_noise_U.append(noise_QU[:, 1, :])

    # Stack : shape (n_sim_per_rank, n_freq, n_pix)
    all_noise_Q = jnp.array(np.stack(all_noise_Q, axis=0))
    all_noise_U = jnp.array(np.stack(all_noise_U, axis=0))

    # Appliquer W en une seule passe sur toutes les réalisations du rank
    noise_stokes_batch = Stokes.from_stokes(Q=all_noise_Q, U=all_noise_U)
    
    # vmap sur la dimension batch — une seule compilation JAX
    W_vmap = jax.vmap(diagonal_central_term, in_axes=0)
    noise_maps_post_compsep_batch = W_vmap(noise_stokes_batch)

    # Boucler sur les résultats pour calculer les spectres
    for i, id_realisation in enumerate(rank_realisation_list):
        noise_comp_dict = {
            "Noise_CMB": np.array([
                np.asarray(noise_maps_post_compsep_batch["cmb"].q[i]),
                np.asarray(noise_maps_post_compsep_batch["cmb"].u[i]),
            ]) * binary_mask,
            "Noise_Dust": np.array([
                np.asarray(noise_maps_post_compsep_batch["dust"].q[i]),
                np.asarray(noise_maps_post_compsep_batch["dust"].u[i]),
            ]) * binary_mask,
        }
        if config.parametric_sep_pars.include_synchrotron:
            noise_comp_dict["Noise_Synch"] = np.array([
                np.asarray(noise_maps_post_compsep_batch["synchrotron"].q[i]),
                np.asarray(noise_maps_post_compsep_batch["synchrotron"].u[i]),
            ]) * binary_mask

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
        for key in noise_Cls:
            if key not in sum_noise_spectra:
                sum_noise_spectra[key] = np.zeros_like(noise_Cls[key])
            sum_noise_spectra[key] += noise_Cls[key]

    # Réduction MPI après la boucle
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
    world, rank, size = get_world()
    parser = argparse.ArgumentParser(description="Noise spectra estimator")
    parser.add_argument("--config", type=Path, required=True, help="config file")
    parser.add_argument("--sim", type=int, default=None, help="process only this simulation index")

    args = parser.parse_args()
    config = Config.load_yaml(args.config)
    manager = DataManager(config)

    workspace_nmt, effective_beam_CMB = init_workspace(config, manager)
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