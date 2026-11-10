import argparse
import faulthandler
import gc
import math
import multiprocessing as mp
import os
import queue as queue_mod
import traceback
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import healpy as hp  # noqa: E402
import megabuster as mb  # noqa: E402
import numpy as np  # noqa: E402
import pymaster as nmt  # noqa: E402
from mpi4py import MPI  # noqa: E402

from megatop import Config, DataManager  # noqa: E402
from megatop.utils import Timer, logger, mask  # noqa: E402
from megatop.utils.binning import load_nmt_binning  # noqa: E402
from megatop.utils.mpi import get_world  # noqa: E402
from megatop.utils.spectra import (  # noqa: E402
    compute_auto_cross_cl_from_maps_dict,
    get_common_beam_wpix,
    get_common_wpix,
    initialize_nmt_workspace,
    limit_namaster_output,
)
from megatop.utils.utils import MemoryUsage  # noqa: E402

def _flatten_pix(a, n_lead: int):
    """Flatten the pixel axes after the first `n_lead` axes (no-op for HEALPix; CAR (ny, nx) -> ny*nx)."""
    a = np.asarray(a)
    return a.reshape(*a.shape[:n_lead], -1)


def _unflatten_pix(a, pix_shape):
    """Inverse of `_flatten_pix` on the last axis (no-op for HEALPix)."""
    a = np.asarray(a)
    return a.reshape(*a.shape[:-1], *pix_shape)


def _union_binary_mask(config: Config, manager: DataManager):
    """Per-experiment binary masks + their union, read via the landscape (HEALPix or CAR)."""
    experiments = set(m.exp_tag for m in config.map_sets)
    binary_mask_per_exp = {
        exp: config.landscape.read_map(manager.path_to_binary_mask(exp)) for exp in experiments
    }
    binary_mask = config.landscape.zeros(()).astype(bool)
    for exp in experiments:
        binary_mask |= binary_mask_per_exp[exp].astype(bool)
    return binary_mask_per_exp, binary_mask


# ---------------------------------------------------------------------------
# Workspace NaMaster : calculé une seule fois (rang 0), écrit sur disque,
# relu par chaque processus fille (un NmtWorkspace n'est pas picklable).
# ---------------------------------------------------------------------------
def init_workspace(config: Config, manager: DataManager):
    nmt_bins = load_nmt_binning(manager)

    # use_beam = config.parametric_sep_pars.megabuster_options.use_beam_operator
    use_beam = False

    # CAR fields carry the geometry's wcs into NaMaster; HEALPix passes None.
    wcs = config.geometry[1] if config.is_car else None

    if use_beam:
        effective_beam_CMB = get_common_wpix(config.nside, config.lmax,  is_car=config.is_car)
    else:
        effective_beam_CMB = get_common_beam_wpix(
            config.pre_proc_pars.common_beam_correction, config.nside, config.lmax, is_car=config.is_car)
        logger.warning(
            "We are only using the CMB effective beam in the noise spectra estimation\n"
            "If you want to use the effective beam for the other components, please update the code"
        )

    analysis_mask = config.landscape.read_map(manager.path_to_joint_analysis_mask)

    with Timer("init-namaster-workspace"):
        workspace = initialize_nmt_workspace(
            nmt_bins=nmt_bins,
            analysis_mask=analysis_mask,
            beam=effective_beam_CMB,
            purify_e=config.map2cl_pars.purify_e,
            purify_b=config.map2cl_pars.purify_b,
            n_iter=config.map2cl_pars.n_iter_namaster,
            lmax=config.lmax,
            wcs=wcs,
        )
    return workspace, effective_beam_CMB


def _load_workspace(path: Path):
    # NaMaster >= 2 : NmtWorkspace.from_file ; versions antérieures : read_from
    if hasattr(nmt.NmtWorkspace, "from_file"):
        return nmt.NmtWorkspace.from_file(str(path))
    w = nmt.NmtWorkspace()
    w.read_from(str(path))
    return w


# ---------------------------------------------------------------------------
# Isolation dans un processus fille (même principe que l'étape compsep)
# ---------------------------------------------------------------------------
def _child_entrypoint(func, args, q):
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    faulthandler.enable(all_threads=True)  # trace Python si segfault dans la fille
    try:
        q.put(("ok", func(*args)))
    except Exception:
        q.put(("err", traceback.format_exc()))


def run_isolated(func, *args, poll_seconds: float = 5.0):
    """Exécute func(*args) dans un processus 'spawn' neuf et renvoie son résultat.

    Contrairement à un simple queue.get(), on surveille la fille : si elle meurt
    (segfault, OOM...) sans rien renvoyer, on lève une erreur au lieu de bloquer
    indéfiniment.
    """
    ctx = mp.get_context("spawn")

    # Retirer les variables MPI pour que la fille ne tente pas de rejoindre
    # le monde MPI du parent (blocage silencieux sous mpirun).
    env_backup = os.environ.copy()
    for key in list(os.environ):
        if key.startswith(("OMPI_", "PMIX_", "PMI_")):
            del os.environ[key]
    try:
        q = ctx.Queue()
        p = ctx.Process(target=_child_entrypoint, args=(func, args, q))
        p.start()  # l'environnement est copié ici
    finally:
        os.environ.clear()
        os.environ.update(env_backup)

    while True:
        try:
            status, payload = q.get(timeout=poll_seconds)
            break
        except queue_mod.Empty:
            if not p.is_alive():
                try:  # la fille a pu écrire juste avant de sortir
                    status, payload = q.get(timeout=1.0)
                    break
                except queue_mod.Empty:
                    raise RuntimeError(
                        f"Processus fille mort sans résultat (exitcode={p.exitcode}; -11 = segfault)"
                    ) from None
    p.join()

    if status == "err":
        raise RuntimeError(f"Échec dans le processus fille:\n{payload}")
    return payload


# ---------------------------------------------------------------------------
# Travail exécuté DANS la fille : aucune communication MPI ici.
# Renvoie la somme des spectres sur les réalisations données,
# ou None si les résultats de compsep sont absents.
# ---------------------------------------------------------------------------
def _compute_noise_sum(
    config: Config,
    manager: DataManager,
    workspace_path: Path,
    effective_beam_CMB,
    id_sim_sky: int | None,
    realisations: list[int],
):
    workspace_nmt = _load_workspace(workspace_path)

    n_sim_noise = config.noise_sim_pars.n_sim

    if not config.parametric_sep_pars.use_megabuster:
        with np.load(manager.get_path_to_compsep_results(sub=id_sim_sky), allow_pickle=True) as f:
            W_maxL = f["W_maxL"]

    binary_mask_per_exp, binary_mask = _union_binary_mask(config, manager)

    # CAR fields carry the geometry's wcs into NaMaster; HEALPix passes None.
    wcs = config.geometry[1] if config.is_car else None

    analysis_mask = config.landscape.read_map(manager.path_to_joint_analysis_mask)
    nmt_bins = load_nmt_binning(manager)

    if config.parametric_sep_pars.use_megabuster:
        with Timer("init-megabuster"):
            logger.warning(
                "Using Megabuster for component separation, make sure to have the correct parameters set in the config file"
            )
            try:
                with np.load(
                    manager.get_path_to_compsep_results(id_sim_sky), allow_pickle=True
                ) as f:
                    parameters_foregrounds_x = f["x"]
                    params_names = f["params"]  # même ordre que "x"
                parameters_dict = {
                    str(name): parameters_foregrounds_x[i] for i, name in enumerate(params_names)
                }
                logger.info(f"Estimated parameters used for noise propagation: {parameters_dict}")
            except FileNotFoundError:
                logger.error(
                    f"Results from comp sep not found for {manager.get_path_to_compsep_results(id_sim_sky)}"
                )
                return None

            if config.parametric_sep_pars.megabuster_options.use_obsmat and config.is_car:
                raise NotImplementedError("CAR: use_obsmat not supported (NEST-ordered HEALPix obsmat).")
            if config.parametric_sep_pars.megabuster_options.use_obsmat:
                obsmat_operator_fname = manager.get_path_list_or_None("suffix_obsmat_scipy")
                if np.any(np.array(obsmat_operator_fname) == Path()):
                    raise ValueError("Not all observation matrix files are provided.")
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

            noisecov_QU_masked = noisecov[:, 1:].copy()
            del noisecov
            for i_m, map_set in enumerate(config.map_sets):
                noisecov_QU_masked[i_m] = mask.apply_binary_mask(
                    noisecov_QU_masked[i_m], binary_mask_per_exp[map_set.exp_tag], unseen=False
                )
            inverse_noisecov_QU_masked = np.zeros_like(noisecov_QU_masked)
            nz = noisecov_QU_masked != 0
            inverse_noisecov_QU_masked[nz] = 1.0 / noisecov_QU_masked[nz]

            if config.pre_proc_pars.use_real_beams:
                real_noisecov = np.load(manager.path_to_real_pixel_noisecov)
                real_noisecov_QU_masked = real_noisecov[:, 1:].copy()
                del real_noisecov
                for i_m, map_set in enumerate(config.map_sets):
                    real_noisecov_QU_masked[i_m] = mask.apply_binary_mask(
                        real_noisecov_QU_masked[i_m],
                        binary_mask_per_exp[map_set.exp_tag],
                        unseen=False,
                    )
                real_inverse_noisecov_QU_masked = np.zeros_like(real_noisecov_QU_masked)
                nz = real_noisecov_QU_masked != 0
                real_inverse_noisecov_QU_masked[nz] = 1.0 / real_noisecov_QU_masked[nz]
            else:
                real_inverse_noisecov_QU_masked = None

            # CAR: per-pixel compsep on flat pixel arrays (no-op for HEALPix)
            inverse_noisecov_QU_masked = _flatten_pix(inverse_noisecov_QU_masked, 2)
            if real_inverse_noisecov_QU_masked is not None:
                real_inverse_noisecov_QU_masked = _flatten_pix(real_inverse_noisecov_QU_masked, 2)
            binary_mask_flat = _flatten_pix(binary_mask, 0)

            megabuster_options = config.parametric_sep_pars.get_megabuster_options_as_dict()
            angles_names = [f"angle_{i}" for i in range(len(config.angle_uncertainty))]

    if (
        config.pre_proc_pars.correct_for_TF and config.parametric_sep_pars.use_harmonic_compsep
    ) and not config.parametric_sep_pars.alm2map:
        logger.info("Computing effective Transfer Function after component separation")
        transfer_freq = []
        for tf_path in manager.get_TF_filenames():
            with np.load(tf_path, allow_pickle=True) as f:
                transfer_freq.append(f["full_tf"])
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

    for id_realisation in realisations:
        MemoryUsage(f"sim {id_sim_sky} / real {id_realisation} ")
        id_real = None if n_sim_noise is None else id_realisation
        logger.info(f"id_realisation = {id_real}")

        if config.noise_sim_pars.DEBUG_save_TRUEnoise_simulations:
            path_true_preproc_noise_maps = manager.get_path_to_preprocessed_TRUE_noise_maps(
                id_sim_sky
            )
            logger.info(f"Loading TRUE noise simulations from {path_true_preproc_noise_maps}")
            noise_freq_maps_preprocessed = np.load(path_true_preproc_noise_maps)
            real_noise_freq_maps_preprocessed = None
        else:
            logger.info("Loading pre-processed noise maps")
            noise_freq_maps_preprocessed = np.load(
                manager.get_path_to_preprocessed_noise_maps(id_real)
            )
            if config.pre_proc_pars.use_real_beams:
                real_noise_freq_maps_preprocessed = np.load(
                    manager.get_path_to_real_preprocessed_noise_maps(id_real)
                )
            else:
                real_noise_freq_maps_preprocessed = None

        pix_shape = noise_freq_maps_preprocessed.shape[2:]
        if not config.parametric_sep_pars.use_megabuster:
            # W_maxL carries a flat pixel axis: flatten the noise maps to match, then restore
            noise_map_post_compsep = np.einsum(
                "ifsp,fsp->isp", W_maxL, _flatten_pix(noise_freq_maps_preprocessed[:, 1:], 2)
            )
            noise_map_post_compsep = _unflatten_pix(noise_map_post_compsep, pix_shape)
        else:
            # megabuster vide ce dictionnaire (pop) : on en crée un neuf à chaque appel
            angles_prior_dict = {
                "angle_central_value": config.angle_central_value,
                "angle_uncertainty": config.angle_uncertainty,
            }

            noise_masked = noise_freq_maps_preprocessed[:, 1:].copy()
            for i_m, map_set in enumerate(config.map_sets):
                noise_masked[i_m] = mask.apply_binary_mask(
                    noise_masked[i_m], binary_mask_per_exp[map_set.exp_tag], unseen=False
                )
            if real_noise_freq_maps_preprocessed is not None:
                real_noise_masked = real_noise_freq_maps_preprocessed[:, 1:].copy()
                for i_m, map_set in enumerate(config.map_sets):
                    real_noise_masked[i_m] = mask.apply_binary_mask(
                        real_noise_masked[i_m], binary_mask_per_exp[map_set.exp_tag], unseen=False
                    )
            else:
                real_noise_masked = None

            noise_masked = _flatten_pix(noise_masked, 2)
            if real_noise_masked is not None:
                real_noise_masked = _flatten_pix(real_noise_masked, 2)

            compsep_results = mb.compsep.perform_compsep(
                config,
                manager,
                first_guess_params=parameters_dict,
                fixed_params={"temp_dust": 20.0},
                sky_map=noise_masked,
                real_sky_map=real_noise_masked,
                frequencies=np.array(config.frequencies),
                invN_matrix=inverse_noisecov_QU_masked,
                real_invN_matrix=real_inverse_noisecov_QU_masked,
                do_minimization=False,
                binary_mask=binary_mask_flat,
                obs_mat_operator=None,
                obsmat_operator_rhs=obsmat_operator_rhs,
                use_calibration_matrix=megabuster_options["use_calibration_matrix"],
                angles_prior_dict=angles_prior_dict,
                angles_names=list(angles_names),
                use_preconditioner_diag=megabuster_options["use_preconditioner_diag"],
                use_preconditioner_pinv=megabuster_options["use_preconditioner_pinv"],
                central_freq_op=central_freq_op,
                matrix_precond=matrix_precond,
                dictionary_parameters_CG={
                    "max_steps_CG": megabuster_options["max_steps_CG"],
                    "tol_CG": megabuster_options["tol_CG"],
                },
                ordering_parameter=angles_names + ["beta_dust", "beta_pl"],
                ordering_component=["cmb", "dust", "synchrotron"],
                use_hessienne=megabuster_options["use_hessienne"],
                use_hmc=megabuster_options["use_hmc"],
                n_warm=megabuster_options["n_warm"],
                n_samples=megabuster_options["n_samples"],
            )
            # Sortie JAX -> vrai tableau numpy contigu et modifiable avant NaMaster
            noise_map_post_compsep = np.array(
                compsep_results.s, dtype=np.float64, copy=True, order="C"
            )
            # back to the native pixel geometry for masking + NaMaster (no-op for HEALPix)
            noise_map_post_compsep = np.ascontiguousarray(
                _unflatten_pix(noise_map_post_compsep, pix_shape)
            )
            del compsep_results, noise_masked, real_noise_masked

        noise_map_post_compsep *= binary_mask

        noise_comp_dict = {
            "Noise_CMB": noise_map_post_compsep[0],
            "Noise_Dust": noise_map_post_compsep[1],
        }
        if config.parametric_sep_pars.include_synchrotron:
            noise_comp_dict["Noise_Synch"] = noise_map_post_compsep[2]

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
            wcs=wcs,
        )

        for key, val in noise_Cls.items():
            val = np.asarray(val)
            if key not in sum_noise_spectra:
                sum_noise_spectra[key] = np.zeros_like(val)
            sum_noise_spectra[key] += val

        del noise_Cls, noise_comp_dict, noise_map_post_compsep
        del noise_freq_maps_preprocessed, real_noise_freq_maps_preprocessed
        gc.collect()
        logger.info(f"Finished realisation {id_realisation}\n")

    return sum_noise_spectra


# ---------------------------------------------------------------------------
# Orchestration côté parent : répartition MPI, une fille par paquet de
# réalisations, réduction, sauvegarde.
# ---------------------------------------------------------------------------
def noise_spectra_estimator(
    config: Config,
    manager: DataManager,
    workspace_path: Path,
    effective_beam_CMB,
    id_sim_sky: int | None = None,
    chunk_size: int = 10,
):
    comm, rank, size = get_world()
    root = 0

    n_sim_noise = config.noise_sim_pars.n_sim
    int_n_sim_noise = (
        1
        if n_sim_noise is None or config.noise_sim_pars.DEBUG_save_TRUEnoise_simulations
        else n_sim_noise
    )
    # La répartition se fait ICI : dans la fille, get_world() verrait size=1.
    rank_realisations = np.array_split(np.arange(int_n_sim_noise), size)[rank]

    local_sum = {}
    failed = False
    error_msg = None
    n_chunks = max(1, math.ceil(len(rank_realisations) / chunk_size))
    try:
        for chunk in np.array_split(rank_realisations, n_chunks):
            if chunk.size == 0:
                continue
            logger.info(f"rank {rank}: sim {id_sim_sky}, réalisations {chunk.tolist()}")
            partial_sum = run_isolated(
                _compute_noise_sum,
                config,
                manager,
                workspace_path,
                effective_beam_CMB,
                id_sim_sky,
                chunk.tolist(),
            )
            if partial_sum is None:  # résultats compsep absents
                failed = True
                break
            for key, val in partial_sum.items():
                local_sum[key] = local_sum[key] + val if key in local_sum else val
    except RuntimeError as e:
        # On ne relève pas tout de suite : les autres rangs attendent le gather.
        failed = True
        error_msg = str(e)
        logger.error(f"rank {rank}, sim {id_sim_sky}: {error_msg}")

    # Réduction : gather de dicts (petits) -> pas de collectives désynchronisées,
    # même si un rang n'a eu aucune réalisation.
    if comm is not None:
        any_failed = comm.allreduce(failed, op=MPI.LOR)
        all_sums = comm.gather(local_sum, root=root)
    else:
        any_failed = failed
        all_sums = [local_sum]

    if any_failed:
        logger.error(f"Sim {id_sim_sky} ignorée : au moins un rang a échoué.")
        return id_sim_sky

    if rank == root:
        total = {}
        for d in all_sums:
            for key, val in d.items():
                total[key] = total[key] + val if key in total else val.copy()

        with np.load(manager.path_to_binning, allow_pickle=True) as f:
            bin_index_lminlmax = f["bin_index_lminlmax"]

        mean_noise_spectra = {key: val / int_n_sim_noise for key, val in total.items()}
        mean_noise_spectra = limit_namaster_output(mean_noise_spectra, bin_index_lminlmax)

        fname = manager.get_path_to_noise_spectra_cross_components(id_sim_sky)
        logger.info(f"Saving estimated noise spectra to {fname}")
        np.savez(fname, **mean_noise_spectra)

    return id_sim_sky

def build_Lambda(A: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """Λ = C A .

    A      : (n_freq, n_comp), matrice de mélange
    angles : (n_freq,), angles de calibration en radians
    Retourne : (2*n_freq, 2*n_comp), lignes (ν, E/B), colonnes (comp, E/B)
    """
    c, s = np.cos(2 * angles), np.sin(2 * angles)
    R = np.stack([np.stack([c, -s], axis=-1),
                  np.stack([s,  c], axis=-1)], axis=-2)   # (n_freq, 2, 2) : rotation C par fréquence
    n_f, n_c = A.shape
    Lam = np.einsum("fc,fxy->fxcy", A, R)                 # (n_freq, 2, n_comp, 2)
    return Lam.reshape(2 * n_f, 2 * n_c)


def analytic_noise_cov(Lambda_EB: np.ndarray, nl: np.ndarray) -> np.ndarray:
    """(Λ^t N_ell^{-1} Λ)^{-1} complète, en (composante, E/B).

    Lambda_EB : (2*n_freq, 2*n_comp)
    nl        : (n_freq, 3, n_ell) avec l'axe 1 = (0, EE, BB) ; N^EB supposé nul
    Retourne  : (n_ell, n_comp, 2, n_comp, 2), indices [ell, comp_i, E/B, comp_j, E/B]
    """
    inv_nl = np.stack([1 / nl[:, 1, :], 1 / nl[:, 2, :]], axis=1)   # (n_freq, 2, n_ell)
    inv_nl = inv_nl.reshape(-1, nl.shape[-1])                       # (2*n_freq, n_ell)
    M = np.einsum("ia,il,ib->lab", Lambda_EB, inv_nl, Lambda_EB)    # (n_ell, 2n_comp, 2n_comp)
    Minv = np.linalg.inv(M)
    n_c = Lambda_EB.shape[1] // 2
    return Minv.reshape(-1, n_c, 2, n_c, 2)

def mixing_matrix_array(frequencies, params, dust_nu0=150.0, synchrotron_nu0=150.0) -> np.ndarray:
    """A (n_freq, n_comp) construite avec la même fonction que la comp-sep, sur un pixel factice."""
    import jax.numpy as jnp
    from furax.obs.stokes import Stokes
    from megabuster.mixingmatrix import create_MixingMatrixOperator

    n_f = len(frequencies)
    dummy = Stokes.from_stokes(q=jnp.zeros((n_f, 1)), u=jnp.zeros((n_f, 1)))
    in_structure_sed = dummy.structure_for((dummy.shape[1],))   # même construction que perform_compsep
    A_op = create_MixingMatrixOperator(
        frequencies, params, in_structure_sed,
        dust_nu0=dust_nu0, synchrotron_nu0=synchrotron_nu0, patch_indices=None,
    )
    # block_leaves dans l'ordre cmb, dust, synchrotron
    return np.stack(
        [np.asarray(leaf._diagonal).reshape(n_f, -1)[:, 0] for leaf in A_op.block_leaves], axis=1
    )

def compute_noise_nl_like_mc(
    config: Config,
    manager: DataManager,
    workspace_path: Path,
    effective_beam_CMB,
    n_real: int | None = None,
) -> np.ndarray:
    """N_ell de bruit par fréquence, calculés avec le même estimateur que les spectres MC.

    Même masquage des cartes (binaire par expérience puis union), même workspace
    (masque d'analyse joint, purification, beam effectif), même binning final.
    Mis en cache sur disque (indépendant de la simu de ciel).

    Retourne : (n_freq, 4, n_bins) avec l'axe 1 = (EE, EB, BE, BB), déconvolué du beam effectif.
    """
    cache_path = Path(workspace_path).parent / "nl_noise_freq_like_mc.npy"
    if cache_path.exists():
        logger.info(f"Loading cached noise N_ell (MC-like estimator) from {cache_path}")
        return np.load(cache_path)

    workspace_nmt = _load_workspace(workspace_path)
    analysis_mask = config.landscape.read_map(manager.path_to_joint_analysis_mask)
    binary_mask_per_exp, binary_mask = _union_binary_mask(config, manager)
    wcs = config.geometry[1] if config.is_car else None

    with np.load(manager.path_to_binning, allow_pickle=True) as f:
        bin_idx = f["bin_index_lminlmax"]

    n_sim_noise = config.noise_sim_pars.n_sim
    if n_sim_noise is None:
        realisations = [None]
    else:
        realisations = range(n_sim_noise if n_real is None else min(n_real, n_sim_noise))

    n_f = len(config.map_sets)
    acc = None
    for id_real in realisations:
        maps = np.load(manager.get_path_to_preprocessed_noise_maps(id_real))[:, 1:].copy()  # (n_freq, 2, npix) Q, U
        for i_m, map_set in enumerate(config.map_sets):
            maps[i_m] = mask.apply_binary_mask(maps[i_m], binary_mask_per_exp[map_set.exp_tag], unseen=False)
        maps *= binary_mask

        nl_real = []
        for i in range(n_f):
            cls = compute_auto_cross_cl_from_maps_dict(
                maps_dict={f"f{i}": maps[i]},
                analysis_mask=analysis_mask,
                workspace=workspace_nmt,
                beam=effective_beam_CMB,
                n_iter=config.map2cl_pars.n_iter_namaster,
                lmax=config.lmax,
                purify_b=config.map2cl_pars.purify_b,
                purify_e=config.map2cl_pars.purify_e,
                wcs=wcs,
            )
            nl_real.append(np.asarray(cls[f"f{i}xf{i}"]))
        nl_real = np.array(nl_real)                                   # (n_freq, 4, n_bins_full)
        acc = nl_real if acc is None else acc + nl_real
        logger.info(f"Noise N_ell (MC-like): realisation {id_real} done")

    nl = (acc / len(realisations))[..., bin_idx]                      # mêmes bins que limit_namaster_output
    np.save(cache_path, nl)
    logger.info(f"Saved noise N_ell (MC-like estimator) to {cache_path}")
    return nl

def pixel_weight_kernel(config: Config, manager: DataManager, Lambda_EB: np.ndarray) -> np.ndarray:
    """Noyau G des poids pixel de la comp-sep, moyenné sur le masque d'analyse.

    W_p = (Λ^t N_p^{-1} Λ)^{-1} Λ^t N_p^{-1} avec N_p la variance par pixel (comme megabuster),
    G[a, b, i] = Σ_p m_p² W_p[a, i] W_p[b, i] / Σ_p m_p².

    Retourne : (2*n_comp, 2*n_comp, 2*n_freq)
    """
    path_cov = (
        manager.path_to_real_pixel_noisecov
        if config.pre_proc_pars.use_real_beams
        else manager.path_to_pixel_noisecov
    )
    # CAR: flatten pixel axes to a single axis p (no-op for HEALPix)
    noisecov = _flatten_pix(np.load(path_cov), 2)                   # (n_freq, 3, npix)
    analysis_mask = _flatten_pix(config.landscape.read_map(manager.path_to_joint_analysis_mask), 0)

    # Pixel solid angle: constant for HEALPix (cancels in the normalised average),
    # varies as cos(dec) for CAR so the mask average must be area-weighted.
    if config.is_car:
        from pixell import enmap

        pix_area = _flatten_pix(enmap.pixsizemap(*config.geometry), 0)
    else:
        pix_area = np.ones(analysis_mask.size)

    n_f = len(config.map_sets)
    inv_var = np.zeros((n_f, noisecov.shape[-1]))
    for i_m, map_set in enumerate(config.map_sets):
        binary = _flatten_pix(
            config.landscape.read_map(manager.path_to_binary_mask(map_set.exp_tag)), 0
        ).astype(bool)
        var_Q = noisecov[i_m, 1]
        ok = binary & (var_Q > 0)
        inv_var[i_m, ok] = 1.0 / var_Q[ok]

    # Pixels utilisés : dans le masque d'analyse, avec assez de fréquences observées pour inverser
    n_c = Lambda_EB.shape[1] // 2
    pix = np.where((analysis_mask > 0) & ((inv_var > 0).sum(axis=0) >= n_c))[0]
    m2 = analysis_mask[pix] ** 2 * pix_area[pix]

    Ninv = np.repeat(inv_var[:, pix].T, 2, axis=1)                 # (n_pix, 2*n_freq), même valeur pour E et B
    LtN = Lambda_EB.T[None, :, :] * Ninv[:, None, :]               # (n_pix, 2n_comp, 2n_freq)
    M = LtN @ Lambda_EB                                            # (n_pix, 2n_comp, 2n_comp)
    W_p = np.linalg.solve(M, LtN)                                  # (n_pix, 2n_comp, 2n_freq)

    G = np.einsum("p,pai,pbi->abi", m2, W_p, W_p) / m2.sum()

    # Diagnostic : variation des poids CMB-B sur le ciel
    w_cmb_B = W_p[:, 1, :]
    rel_spread = np.std(w_cmb_B, axis=0) / (np.abs(np.mean(w_cmb_B, axis=0)) + 1e-30)
    logger.info(f"Poids CMB (B) moyens par (freq, E/B) : {np.mean(w_cmb_B, axis=0)}")
    logger.info(f"Dispersion relative des poids sur le ciel : {rel_spread}")
    return G


def noise_cov_from_kernel(G: np.ndarray, nl: np.ndarray) -> np.ndarray:
    """C_ell = Σ_i N_ell,i G[:, :, i] (ligne du haut de l'éq. 22 avec les poids pixel).

    G  : (2*n_comp, 2*n_comp, 2*n_freq)
    nl : (n_freq, 3, n_ell) avec l'axe 1 = (0, EE, BB)
    Retourne : (n_ell, n_comp, 2, n_comp, 2)
    """
    N = np.stack([nl[:, 1, :], nl[:, 2, :]], axis=1).reshape(-1, nl.shape[-1])   # (2*n_freq, n_ell)
    cov = np.einsum("abi,il->lab", G, N)
    n_c = G.shape[0] // 2
    return cov.reshape(-1, n_c, 2, n_c, 2)

def analytic_noise_spectra(
    config: Config, manager: DataManager, effective_beam_CMB, id_sim_sky,
    workspace_path: Path, n_real_nl: int | None = None,
):
    """N_ell analytique du CMB reconstruit, binné comme les spectres MC, et sauvegardé."""
    with np.load(manager.get_path_to_compsep_results(id_sim_sky), allow_pickle=True) as f:
        x, names = f["x"], [str(n) for n in f["params"]]
    params = {n: float(v) for n, v in zip(names, x)}
    params.setdefault("temp_dust", 20.0)            # fixé comme dans l'appel à perform_compsep

    freqs = np.array(config.frequencies)
    n_f = len(freqs)

    use_cal = config.parametric_sep_pars.get_megabuster_options_as_dict()["use_calibration_matrix"]
    for i in range(n_f):
        if f"angle_{i}" not in params:
            logger.warning(f"angle_{i} absent des résultats de la comp-sep, valeur centrale de la config utilisée")
            params[f"angle_{i}"] = float(config.angle_central_value[i])
    angles = np.array([params[f"angle_{i}"] for i in range(n_f)]) if use_cal else np.zeros(n_f)
    logger.info(f"[sim {id_sim_sky}] angles utilisés (deg) : {np.degrees(angles)}")
    
    A = mixing_matrix_array(freqs, params)

        # N_ell par fréquence avec le même estimateur que le MC (masque joint, purification, beam effectif)
    nl4 = compute_noise_nl_like_mc(config, manager, workspace_path, effective_beam_CMB, n_real=n_real_nl)
    # Format attendu en aval : (n_freq, 3, n_bins) = (0, EE, BB)
    nl_deconv = np.stack([np.zeros_like(nl4[:, 0]), nl4[:, 0], nl4[:, 3]], axis=1)

    Lambda_EB = build_Lambda(A, angles)
    G = pixel_weight_kernel(config, manager, Lambda_EB)
    cov = noise_cov_from_kernel(G, nl_deconv)              # poids pixel de la comp-sep (ligne du haut)
    cov_opt = analytic_noise_cov(Lambda_EB, nl_deconv) 

    # Même format que les spectres MC : clés "Noise_AxNoise_B", tableaux (4, n_bins) = [EE, EB, BE, BB]
    comp_names = ["Noise_CMB", "Noise_Dust", "Noise_Synch"][: cov.shape[1]]
    analytic_spectra = {}
    for i in range(cov.shape[1]):
        for j in range(i, cov.shape[1]):
            analytic_spectra[f"{comp_names[i]}x{comp_names[j]}"] = np.stack([
                cov[:, i, 0, j, 0],   # EE
                cov[:, i, 0, j, 1],   # EB
                cov[:, i, 1, j, 0],   # BE
                cov[:, i, 1, j, 1],   # BB
            ])

    fname = Path(manager.get_path_to_noise_spectra_cross_components(id_sim_sky))
    fname = fname.with_name(fname.stem + "_analytic.npz")
    np.savez(fname, **analytic_spectra, A=A, angles=angles)
    logger.info(f"Saved analytic noise spectra to {fname}")
    return analytic_spectra



def compare_noise_spectra(config: Config, manager: DataManager, id_sim_sky, make_plot: bool = True):
    """Compare les spectres de bruit MC et analytiques pour une simu de ciel.

    Écart normalisé : Delta = (C_MC - C_an) / sqrt(C_an[ii, X] * C_an[jj, Y]),
    soit C_MC / C_an - 1 pour les auto-spectres EE et BB.
    Retourne un dict {clé: {spectre: (moyenne, min, max) de Delta}}.
    """
    path_mc = Path(manager.get_path_to_noise_spectra_cross_components(id_sim_sky))
    path_an = path_mc.with_name(path_mc.stem + "_analytic.npz")
    for p in (path_mc, path_an):
        if not p.exists():
            logger.error(f"Fichier introuvable : {p}")
            return None

    with np.load(path_mc) as f:
        mc = {k: f[k] for k in f.files}
    with np.load(path_an) as f:
        an = {k: f[k] for k in f.files if k not in ("A", "angles")}

    keys = [k for k in an if k in mc]
    missing = [k for k in an if k not in mc]
    if missing:
        logger.warning(f"Clés analytiques absentes du fichier MC : {missing} (clés MC : {list(mc)})")
    if not keys:
        logger.error("Aucune clé commune entre MC et analytique.")
        return None

    nmt_bins = load_nmt_binning(manager)
    with np.load(manager.path_to_binning, allow_pickle=True) as f:
        bin_idx = f["bin_index_lminlmax"]
    ell = nmt_bins.get_effective_ells()[bin_idx]

    spec_names = ["EE", "EB", "BE", "BB"]
    pol_of_spec = {"EE": (0, 0), "EB": (0, 1), "BE": (1, 0), "BB": (1, 1)}   # (pol_i, pol_j), 0 = EE, 3 = BB
    auto_index = {0: 0, 1: 3}                                                # E -> EE, B -> BB

    summary = {}
    deltas = {}
    for key in keys:
        comp_i, comp_j = key.split("x")
        auto_i, auto_j = f"{comp_i}x{comp_i}", f"{comp_j}x{comp_j}"
        summary[key], deltas[key] = {}, {}
        for s, name in enumerate(spec_names):
            pi, pj = pol_of_spec[name]
            norm = np.sqrt(np.abs(an[auto_i][auto_index[pi]] * an[auto_j][auto_index[pj]]))
            delta = (mc[key][s] - an[key][s]) / norm
            deltas[key][name] = delta
            summary[key][name] = (float(np.mean(delta)), float(np.min(delta)), float(np.max(delta)))
            logger.info(
                f"[sim {id_sim_sky}] {key} {name} : Delta moyen = {summary[key][name][0]:+.3e} "
                f"(min {summary[key][name][1]:+.3e}, max {summary[key][name][2]:+.3e})"
            )

    if make_plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(
            2 * len(keys), 4, figsize=(16, 5 * len(keys)), sharex=True,
            gridspec_kw={"height_ratios": [2, 1] * len(keys)},
        )
        axes = np.atleast_2d(axes)
        for k, key in enumerate(keys):
            for s, name in enumerate(spec_names):
                ax, ax_d = axes[2 * k, s], axes[2 * k + 1, s]
                ax.plot(ell, mc[key][s], "o", ms=3, label="MC")
                ax.plot(ell, an[key][s], "-", label="analytique")
                if name in ("EE", "BB") and key.split("x")[0] == key.split("x")[1]:
                    ax.set_yscale("log")
                ax.set_title(f"{key} {name}", fontsize=9)
                ax_d.axhline(0, color="k", lw=0.5)
                ax_d.plot(ell, deltas[key][name], "o-", ms=3)
                ax_d.set_ylabel(r"$\Delta$")
                if k == 0 and s == 0:
                    ax.legend(fontsize=8)
        for ax in axes[-1]:
            ax.set_xlabel(r"$\ell$")
        fig.tight_layout()
        path_fig = path_mc.with_name(path_mc.stem + "_comparison.png")
        fig.savefig(path_fig, dpi=120)
        plt.close(fig)
        logger.info(f"Figure de comparaison sauvegardée : {path_fig}")

    return summary

def main():
    world, rank, size = get_world()
    parser = argparse.ArgumentParser(description="Noise spectra estimator")
    parser.add_argument("--config", type=Path, required=True, help="config file")
    parser.add_argument("--sim", type=int, default=None, help="process only this simulation index")
    #parser.add_argument(
    #    "--chunk_size",
    #    type=int,
    #    default=10,
    #    help="nombre de réalisations de bruit traitées par processus fille",
    #)
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=10,
        help="nombre de réalisations de bruit traitées par processus fille",
    )
    parser.add_argument("--analytic_only", action="store_true",
                        help="calcule seulement le bruit analytique, sans la boucle MC")
    parser.add_argument("--compare", action="store_true",
                        help="compare les spectres de bruit MC et analytiques (avec --analytic_only)")
    parser.add_argument("--n_real_nl", type=int, default=None,
                        help="nombre de réalisations de bruit pour estimer les N_ell par fréquence (défaut : toutes)")
    args = parser.parse_args()

    config = Config.load_yaml(args.config)
    manager = DataManager(config)

    if rank == 0:
        manager.dump_config()
        manager.create_output_dirs(config.map_sim_pars.n_sim, config.noise_sim_pars.n_sim)

    # Workspace calculé une seule fois puis partagé via le disque
    # TODO: adapter l'emplacement si ce dossier n'est pas un dossier de sortie
    workspace_path = Path(manager.path_to_binning).parent / "nmt_workspace_noise_spectra.fits"
    if rank == 0:
        workspace_nmt, effective_beam_CMB = init_workspace(config, manager)
        workspace_nmt.write_to(str(workspace_path))
        del workspace_nmt
    else:
        effective_beam_CMB = None
    if world is not None:
        effective_beam_CMB = world.bcast(effective_beam_CMB, root=0)
        world.Barrier()

    common = dict(
        config=config,
        manager=manager,
        workspace_path=workspace_path,
        effective_beam_CMB=effective_beam_CMB,
        chunk_size=args.chunk_size,
    )

    if args.analytic_only:
        if rank == 0:
            sims = [args.sim] if args.sim is not None else (
                [None] if config.map_sim_pars.n_sim == 0 else range(config.map_sim_pars.n_sim))
            for i in sims:
                analytic_noise_spectra(config, manager, effective_beam_CMB, i, workspace_path, args.n_real_nl)
                if args.compare:
                    compare_noise_spectra(config, manager, i)
        return

    if args.sim is not None:
        noise_spectra_estimator(id_sim_sky=args.sim, **common)
        return

    n_sim_sky = config.map_sim_pars.n_sim
    if n_sim_sky == 0:
        noise_spectra_estimator(id_sim_sky=None, **common)
    else:
        for i in range(n_sim_sky):
            result = noise_spectra_estimator(id_sim_sky=i, **common)
            logger.info(
                f"Finished noise spectra estimation for sky simulation {result + 1}/{n_sim_sky}"
            )


if __name__ == "__main__":
    main()