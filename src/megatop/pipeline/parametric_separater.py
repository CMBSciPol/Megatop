import argparse
from functools import partial
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

from inspect import isfunction  # noqa: E402

import fgbuster as fg  # noqa: E402
import healpy as hp  # noqa: E402
import megabuster as mb  # noqa: E402
import numpy as np  # noqa: E402
from fgbuster.component_model import CMB, Dust, Synchrotron  # noqa: E402
from fgbuster.mixingmatrix import MixingMatrix  # noqa: E402
from fgbuster.separation_recipes import _format_alms  # noqa: E402
from furax_cs import SOLVER_NAMES  # noqa: E402
from mpi4py.futures import MPICommExecutor  # noqa: E402

from megatop import Config, DataManager  # noqa: E402
from megatop.utils import Timer, logger, mask, passband  # noqa: E402
from megatop.utils.compsep import set_alm_tozero_below_lmin  # noqa: E402
from megatop.utils.mpi import get_world  # noqa: E402
from megatop.utils.utils import MemoryUsage  # noqa: E402


def _test_N_alm_format(N_alm):
    inv_N_alm = 1 / N_alm

    # format to have real representation of alm covariance
    inv_N_alm_real = _format_alms(inv_N_alm.astype(np.complex128))

    shape_N_alm_freq_diag = inv_N_alm_real.shape
    shape_N_alm_freq_diag += ((inv_N_alm_real.shape[-1]),)
    inv_N_alm_freq_diag = np.zeros(shape_N_alm_freq_diag)
    for i in range(inv_N_alm_real.shape[-1]):
        inv_N_alm_freq_diag[..., i, i] = inv_N_alm_real[..., i]

    inv_N_alm_freq_diag[np.where(np.isinf(inv_N_alm_freq_diag))] = 0

    return inv_N_alm_freq_diag


def get_and_format_inv_Nl(manager: DataManager, config: Config):
    """
    Loads noise Cl estimated from noise map simulations.
    Makes sure that there is a value for each ell by adding concatenating 0 up to ell_min
    Naively inverts the noise spectra (1/N_ell) to get inverse noise covariance needed for componenet separation.
    """

    # Importing noise Cl computer in pixel_noisecov_estimater.py
    # Here we use the cl_unbinned from namaster wich is C_ell instead of C_bin
    # Each C_ell in a bin is equal to C_bin
    # Cl_from_maps is a 3D array with shape (n_freq, n_auto_spectra, n_ell)
    # Where auto spectra are TT, EE, BB and TT is left null
    # TODO: use full auto-cross spectra?
    Cl_from_maps = np.load(manager.path_to_nl_noisecov_unbinned)
    # add 0 for first bins in the last dimension (ell)
    Cl_from_maps = np.pad(
        Cl_from_maps,
        ((0, 0), (0, 0), (config.parametric_sep_pars.harmonic_lmin, 0)),
        mode="constant",
        constant_values=0,
    )
    inv_Cl_from_maps = np.zeros_like(Cl_from_maps)

    inv_Cl_from_maps[:, 1:, config.parametric_sep_pars.harmonic_lmin + 1 :] = (
        1 / Cl_from_maps[:, 1:, config.parametric_sep_pars.harmonic_lmin + 1 :]
    )

    inv_Cl_from_maps_diag = np.zeros(
        (
            Cl_from_maps.shape[0],
            Cl_from_maps.shape[0],
            Cl_from_maps.shape[1],
            Cl_from_maps.shape[2],
        )
    )
    for i in range(inv_Cl_from_maps_diag.shape[0]):
        inv_Cl_from_maps_diag[i, i] = inv_Cl_from_maps[i]

    return inv_Cl_from_maps_diag.T


def harmonic_comp_sep_interface(manager: DataManager, config: Config, id_sim: int | None = None):
    timer = Timer()
    timer.start("do-compsep")
    instrument = {"frequency": config.frequencies}
    if config.parametric_sep_pars.include_synchrotron:
        components = [
            CMB(),
            Dust(150.0, temp=20.0),
            Synchrotron(150.0),
        ]  # TODO move default to config
    else:
        components = [CMB(), Dust(150.0, temp=20.0)]  # TODO move default to config

    # get the 'options' through the appropriate method which returns a dict
    options = config.parametric_sep_pars.get_minimize_options_as_dict()
    tol = config.parametric_sep_pars.minimize_tol
    method = config.parametric_sep_pars.minimize_method

    # If put to 0, I don't think they weigh on the outcome but it slows the process down and can result in warnings/errors
    binary_mask = hp.read_map(manager.path_to_binary_mask)  # .astype(bool)

    invN = get_and_format_inv_Nl(manager, config)
    invNlm = None

    # cov_alm = np.load(manager.path_to_noisecov_alm)
    # invNlm = _test_N_alm_format(cov_alm)
    # invN = None

    instrument["fwhm"] = [None] * 6  # we don't correct for the beam inside the harmonic compsep
    std_instr = fg.observation_helpers.standardize_instrument(instrument)

    with Timer("load-alms"):
        preproc_alms_fname = manager.get_path_to_preprocessed_alms(sub=id_sim)
        logger.debug(f"Loading input maps from {preproc_alms_fname}")
        data_alms = np.load(preproc_alms_fname)

    data_alms_lmin = set_alm_tozero_below_lmin(
        data_alms.copy(), config.parametric_sep_pars.harmonic_lmin
    )
    # import IPython; IPython.embed()
    res = fg.separation_recipes.harmonic_comp_sep(
        components,
        std_instr,
        data_alms_lmin,
        config.nside,
        config.parametric_sep_pars.harmonic_lmax - 1,
        invN=invN,
        invNlm=invNlm,
        mask=None,
        data_is_alm=True,
        options=options,
        tol=tol,
        method=method,
    )
    res.s_alm = res.s

    A = MixingMatrix(*components)
    A_ev = A.evaluator(np.array(instrument["frequency"]))
    A_maxL = A_ev(res.x)
    res.A_maxL = A_maxL

    AtNA_ell = np.einsum("cf,lmfn,nk->lmck", A_maxL.T, invN, A_maxL)
    invAtNA_ell = np.zeros_like(AtNA_ell)
    invAtNA_ell[config.parametric_sep_pars.harmonic_lmin + 1 :, 1:] = np.linalg.inv(
        AtNA_ell[config.parametric_sep_pars.harmonic_lmin + 1 :, 1:]
    )
    res.invAtNA_ell = invAtNA_ell.T[:, :, 1:]  # removing TT
    W_maxL_ell = np.einsum(
        "ckml, kf, lmfn -> cnml", res.invAtNA_ell, A_maxL.T, invN[:, 1:]
    )  # removing TT
    res.W_maxL_ell = W_maxL_ell

    logger.info("Computing W matrix in PIXEL space from harmonic compsep outputs")
    logger.info("Importing pixel based covariance")
    with Timer("load-covmat"):
        noisecov_fname = manager.path_to_pixel_noisecov
        logger.debug(f"Loading covmat from {noisecov_fname}")
        noisecov = np.load(noisecov_fname)

    noisecov_TQU_masked = mask.apply_binary_mask(noisecov, binary_mask, unseen=True)
    noisecov_QU_masked = noisecov_TQU_masked[:, 1:]

    AtNA = np.einsum("cf, fsp, fk->cksp", A_maxL.T, 1 / noisecov_QU_masked, A_maxL)
    res.invAtNA_map = np.linalg.inv(AtNA.T).T
    res.invAtNA_alm = res.invAtNA
    res.invAtNA = res.invAtNA_map

    W_maxL = np.einsum("ijsp, jf, fsp -> ifsp", res.invAtNA[:, :], A_maxL.T, 1 / noisecov_QU_masked)
    res.W_maxL = W_maxL
    # import IPython; IPython.embed()

    # cov_alm = np.load(manager.path_to_noisecov_alm)
    # inv_cov_alm = 1/cov_alm
    # shape_N_alm_freq_diag = (inv_cov_alm.shape[0]),
    # shape_N_alm_freq_diag += inv_cov_alm.shape
    # # inv_cov_alm[np.where(np.isinf(inv_cov_alm))] = 0
    # AtNA_alm_native = np.einsum("cf,fsl,fk->lsck", A_maxL.T, inv_cov_alm, A_maxL)
    # inv_AtNA_alm_native = np.linalg.inv(AtNA_alm_native)
    # inv_AtNA_alm_native[np.where(np.isinf(inv_AtNA_alm_native))] = 0
    # inv_AtNA_alm_native[np.where(np.isnan(inv_AtNA_alm_native))] = 0
    # W_maxL_alm = np.einsum("lsck,cf,fsl->cfsl", inv_AtNA_alm_native, A_maxL.T, inv_cov_alm)
    # W_maxL_alm[np.where(np.isnan(W_maxL_alm))] = 0

    # shape_TEB = np.array(W_maxL_alm.shape)
    # shape_TEB[-2] += 1  # adding T
    # W_maxL_alm_TEB = np.zeros(shape_TEB)
    # W_maxL_alm_TEB[:, :, 1:, :] = W_maxL_alm

    if config.parametric_sep_pars.alm2map:
        logger.info(
            "Harmonic Compsep: Computing component map from output alms, this might induce some edge effect..."
        )
        res.s = np.array(
            [
                hp.alm2map_spin(
                    res.s_alm[i],
                    nside=config.nside,
                    spin=2,
                    lmax=config.parametric_sep_pars.harmonic_lmax - 1,
                )  # lmax=3 * config.nside
                for i in range(res.s_alm.shape[0])
            ]
        )
        # remove binary mask to avoid double application when entering namaster:
        analysis_mask = hp.read_map(manager.path_to_analysis_mask)
        res.s[..., np.where(binary_mask != 0)] /= analysis_mask[np.where(binary_mask != 0)]
    else:
        logger.info("Harmonic Compsep: Computing component maps using W matrix and input maps")
        logger.warning(
            "Beam and Transfer functions handling? "
        )  # TODO: Beam and Transfer functions handling?
        with Timer("load-maps"):
            preproc_maps_fname = manager.get_path_to_preprocessed_maps(sub=id_sim)
            logger.debug(f"Loading input maps from {preproc_maps_fname}")
            freq_maps_preprocessed = np.load(preproc_maps_fname)
        freq_maps_preprocessed_QU_masked = mask.apply_binary_mask(
            freq_maps_preprocessed[:, 1:], binary_mask, unseen=False
        )
        n_comp = W_maxL.shape[0]
        res.s = np.zeros(
            (
                n_comp,
                freq_maps_preprocessed_QU_masked.shape[-2],
                freq_maps_preprocessed_QU_masked.shape[-1],
            )
        )
        for c in range(n_comp):
            res.s[c] = np.einsum(
                "fsp, fsp -> sp",
                W_maxL[c],
                freq_maps_preprocessed_QU_masked,
            )

    logger.info(f"Success: {res.success} -> {res.message}")
    logger.info(f"Spectral parameters {res.params} -> {res.x}")
    timer.stop("do-compsep")

    return res


def weighted_comp_sep(manager: DataManager, config: Config, id_sim: int | None = None):
    with Timer("load-covmat"):
        noisecov_fname = manager.path_to_pixel_noisecov
        logger.debug(f"Loading covmat from {noisecov_fname}")
        noisecov = np.load(noisecov_fname)

    timer = Timer()
    timer.start("do-compsep")

    if config.parametric_sep_pars.passband_int:
        logger.info("Using passband-integration for the component separation step.")
        config.map_sets = passband.passband_constructor(
            config, manager, passband_int=config.parametric_sep_pars.passband_int
        )
        passbands_norm = passband.fgbuster_passband(config.map_sets)
        instrument = {"frequency": passbands_norm}
    else:
        instrument = {"frequency": config.frequencies}
    if config.parametric_sep_pars.include_synchrotron:
        components = [
            CMB(),
            Dust(150.0, temp=20.0),
            Synchrotron(150.0),
        ]  # TODO move default to config
    else:
        components = [CMB(), Dust(150.0, temp=20.0)]  # TODO move default to config

    # get the 'options' through the appropriate method which returns a dict
    options = config.parametric_sep_pars.get_minimize_options_as_dict()
    tol = config.parametric_sep_pars.minimize_tol
    method = config.parametric_sep_pars.minimize_method

    # FGBuster's weighted component separation used hp.UNSEEN to ignore masked pixels
    # If put to 0, I don't think they weigh on the outcome but it slows the process down and can result in warnings/errors
    binary_mask = hp.read_map(manager.path_to_binary_mask)  # .astype(bool)

    with Timer("load-maps"):
        preproc_maps_fname = manager.get_path_to_preprocessed_maps(sub=id_sim)
        logger.debug(f"Loading input maps from {preproc_maps_fname}")
        freq_maps_preprocessed = np.load(preproc_maps_fname)

    freq_maps_preprocessed_QU_masked = mask.apply_binary_mask(
        freq_maps_preprocessed[:, 1:], binary_mask, unseen=True
    )
    noisecov_QU_masked = mask.apply_binary_mask(noisecov[:, 1:], binary_mask, unseen=True)
    res = fg.separation_recipes.weighted_comp_sep(
        components,
        instrument,
        data=freq_maps_preprocessed_QU_masked,
        cov=noisecov_QU_masked,
        options=options,
        tol=tol,
        method=method,
    )

    A = MixingMatrix(*components)
    if config.parametric_sep_pars.passband_int:
        config.map_sets = passband.passband_constructor(
            config, manager, passband_int=config.parametric_sep_pars.passband_int
        )
        passbands_norm = passband.fgbuster_passband(config.map_sets)
        instrument = {"frequency": passbands_norm}
        A_ev = A.evaluator(instrument["frequency"])
    else:
        instrument = {"frequency": config.frequencies}
        A_ev = A.evaluator(np.array(instrument["frequency"]))

    A_maxL = A_ev(res.x)
    res.A_maxL = A_maxL

    # W_maxL = algebra.W(A_maxL, invN=1 / noisecov_QU_masked)
    W_maxL = np.einsum("ijsp, jf, fsp -> ifsp", res.invAtNA[:, :], A_maxL.T, 1 / noisecov_QU_masked)

    res.W_maxL = W_maxL

    logger.info(f"Success: {res.success} -> {res.message}")
    logger.info(f"Spectral parameters {res.params} -> {res.x}")
    timer.stop("do-compsep")

    return res


def megabuster_comp_sep(manager: DataManager, config: Config, id_sim: int | None = None):
    with Timer("load-covmat"):
        noisecov_fname = manager.path_to_pixel_noisecov
        logger.debug(f"Loading covmat from {noisecov_fname}")
        noisecov = np.load(noisecov_fname)

    with Timer("load-maps"):
        preproc_maps_fname = manager.get_path_to_preprocessed_maps(sub=id_sim)
        logger.debug(f"Loading input maps from {preproc_maps_fname}")
        freq_maps_preprocessed = np.load(preproc_maps_fname)

    binary_mask = hp.read_map(manager.path_to_binary_mask)  # .astype(bool)

    with Timer("load-obsmat"):
        obsmat_operator_fname = manager.get_path_list_or_None("suffix_obsmat_scipy")

        if np.all(np.array(obsmat_operator_fname) == Path()):
            # TODO: how to handle this case?
            logger.warning("No observation matrix file provided. Using identity matrix instead.")
            obsmat_operator_rhs = None
        elif np.any(np.array(obsmat_operator_fname) == Path()):
            msg_any_obs = "Not all observation matrix files are provided. Provide either all or none, partial set of observation matrices is not supported. A temporary solution is to provide a identity observation matrix for channels without filtering"
            raise ValueError(msg_any_obs)
        else:
            logger.debug(f"Loading observation matrix from {obsmat_operator_fname}")
            npix = binary_mask.size
            indices_mask = np.arange(npix)[hp.reorder(binary_mask, r2n=True) != 0]
            mask_stacked_nest = np.hstack((indices_mask + npix, indices_mask + 2 * npix))
            MemoryUsage("Memory usage before build_obsmat_operator_from_flattened_matrices()")

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
                "No eigen decomposition file provided for CG. Using identity matrix instead."
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
            matrix_precond = mb.io.load_matrix_precond(path_eigen_decomp_fname, power_diagonal=-1)

    timer = Timer()
    timer.start("do-compsep")

    if config.parametric_sep_pars.passband_int:
        logger.info("Using passband-integration for the component separation step.")
        # config.map_sets = passband.passband_constructor(
        #     config, manager, passband_int=config.parametric_sep_pars.passband_int
        # )
        # passbands_norm = passband.fgbuster_passband(config.map_sets)
        # instrument = {"frequency": passbands_norm}
        msg_passband = "Megabuster does not support passband integration yet. \nPlease use FGBuster for this feature."
        raise NotImplementedError(msg_passband)
    # instrument = {"frequency": config.frequencies}
    if config.parametric_sep_pars.include_synchrotron:
        components = ["cmb", "dust", "synchrotron"]  # TODO move default to config
    else:
        components = ["cmb", "dust"]  # TODO move default to config

    # get the 'options' through the appropriate method which returns a dict
    options = config.parametric_sep_pars.get_minimize_options_as_dict()
    tol = config.parametric_sep_pars.minimize_tol
    method = config.parametric_sep_pars.minimize_method

    megabuster_options = config.parametric_sep_pars.get_megabuster_options_as_dict()
    
    solver_name = megabuster_options["solver_name"]
    assert solver_name in SOLVER_NAMES.__args__, f"Solver name must be one of {SOLVER_NAMES.__args__}."

    # FGBuster's weighted component separation used hp.UNSEEN to ignore masked pixels
    # If put to 0, I don't think they weigh on the outcome but it slows the process down and can result in warnings/errors

    freq_maps_preprocessed_QU_masked = mask.apply_binary_mask(
        freq_maps_preprocessed[:, 1:], binary_mask, unseen=False
    )
    noisecov_QU_masked = mask.apply_binary_mask(noisecov[:, 1:], binary_mask, unseen=False)
    inverse_noisecov_QU_masked = np.zeros_like(noisecov_QU_masked)
    inverse_noisecov_QU_masked[noisecov_QU_masked != 0] = (
        1.0 / noisecov_QU_masked[noisecov_QU_masked != 0]
    )

    max_iter = options["maxiter"]# if method != "TNC" else options["maxfun"]
    # import IPython; IPython.embed()
    res = mb.compsep.perform_compsep(
        first_guess_params={"beta_dust": np.array(1.54), "beta_pl": np.array(-3.0)},
        fixed_params={"temp_dust": 20.0},
        sky_map=freq_maps_preprocessed_QU_masked,
        frequencies=np.array(config.frequencies),
        invN_matrix=inverse_noisecov_QU_masked,
        do_minimization=True,
        binary_mask=binary_mask,
        obs_mat_operator=None,
        obsmat_operator_rhs=obsmat_operator_rhs,
        use_preconditioner_diag=megabuster_options["use_preconditioner_diag"],
        use_preconditioner_pinv=megabuster_options["use_preconditioner_pinv"],
        central_freq_op=central_freq_op,
        matrix_precond=matrix_precond,
        dictionary_parameters_minimization={
            "max_iter": max_iter, 
            "tol": tol,
            'solver_name': solver_name,
            'options': megabuster_options.get("minimizer_options", dict()),
        },
        dictionary_parameters_CG={
            "max_steps_CG": megabuster_options["max_steps_CG"],
            "tol_CG": megabuster_options["tol_CG"],
        },
        ordering_parameter=["beta_dust", "beta_pl"],
        ordering_component=components,
        dust_nu0=150.0,
        synchrotron_nu0=150.0,
    )

    logger.info(f"Success: {res.success} -> {res.message}")
    logger.info(f"Spectral parameters {res.params} -> {res.x}")
    timer.stop("do-compsep")

    return res


def save_compsep_results(manager: DataManager, config: Config, res, id_sim: int | None = None):
    path = manager.get_path_to_components(sub=id_sim)
    path.mkdir(parents=True, exist_ok=True)
    fname_results = manager.get_path_to_compsep_results(sub=id_sim)
    if config.parametric_sep_pars.use_harmonic_compsep:
        fname_compalms = manager.get_path_to_components_alms(sub=id_sim)
    fname_compmaps = manager.get_path_to_components_maps(sub=id_sim)

    res_dict = {}
    for attr in dir(res):
        if (not attr.startswith("__") and attr != "s") and not isfunction(
            getattr(res, attr)
        ):  # remove component maps to avoid saving twice.
            # if isinstance(getattr(res, attr), callable):
            #     continue # Skip callable attributes
            res_dict[attr] = getattr(res, attr)

    # if type(res_dict["W_maxL"]) is not np.ndarray:
    #     # If W_maxL is a function, we can't pickle it, so we remove it
    #     logger.warning("W_maxL is a function, removing it from the results dictionary.")
    #     res_dict.pop("W_maxL")

    # Saving result dict
    logger.info(f"Saving compsep results to {fname_results}")
    np.savez(fname_results, **res_dict)

    if config.parametric_sep_pars.use_harmonic_compsep:
        # Saving component alms
        logger.info(f"Saving component alms to {fname_compalms}")
        np.save(fname_compalms, res.s_alm)

    # Saving component maps
    logger.info(f"Saving component maps to {fname_compmaps}")
    np.save(fname_compmaps, res.s)


def compsep_and_save(config: Config, manager: DataManager, id_sim: int | None = None):
    # import IPython; IPython.embed()
    with Timer("weighted-compsep"):
        if config.parametric_sep_pars.use_harmonic_compsep:
            res = harmonic_comp_sep_interface(manager, config, id_sim=id_sim)
        elif config.parametric_sep_pars.use_megabuster:
            res = megabuster_comp_sep(manager, config, id_sim=id_sim)
        else:
            res = weighted_comp_sep(manager, config, id_sim=id_sim)
    save_compsep_results(manager, config, res, id_sim=id_sim)
    return id_sim


def main():
    parser = argparse.ArgumentParser(description="Component separation")
    parser.add_argument("--config", type=Path, required=True, help="config file")

    args = parser.parse_args()
    config = Config.load_yaml(args.config)
    manager = DataManager(config)

    world, rank, size = get_world()
    if rank == 0:
        manager.dump_config()

    n_sim_sky = config.map_sim_pars.n_sim
    if n_sim_sky == 0:  # No sky simulations: run preprocessing on the real data
        compsep_and_save(config, manager, id_sim=None)
    else:
        with MPICommExecutor() as executor:
            if executor is not None:
                logger.info(f"Distributing work to {executor.num_workers} workers")
                func = partial(compsep_and_save, config, manager)
                for result in executor.map(func, range(n_sim_sky), unordered=True):
                    logger.info(f"Finished component separation on map {result + 1} / {n_sim_sky}")


if __name__ == "__main__":
    main()
