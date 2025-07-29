import argparse
from functools import partial
from pathlib import Path

import fgbuster as fg
import megabuster as mb
import healpy as hp
import numpy as np
from fgbuster.component_model import CMB, Dust, Synchrotron
from fgbuster.mixingmatrix import MixingMatrix
# from mpi4py.futures import MPICommExecutor

from megatop import Config, DataManager
from megatop.utils import Timer, logger, mask, passband
from megatop.utils.mpi import get_world


def weighted_comp_sep(manager: DataManager, config: Config, id_sim: int | None = None):
    with Timer("load-covmat"):
        noisecov_fname = manager.path_to_pixel_noisecov
        logger.debug(f"Loading covmat from {noisecov_fname}")
        noisecov = np.load(noisecov_fname)

    with Timer("load-maps"):
        preproc_maps_fname = manager.get_path_to_preprocessed_maps(sub=id_sim)
        logger.debug(f"Loading input maps from {preproc_maps_fname}")
        freq_maps_preprocessed = np.load(preproc_maps_fname)

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

    A_maxL = A_ev(res.x)  # pyright: ignore[reportCallIssue]
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
        obsmat_cg_fname = manager.get_path_to_obsmat_cg()
        if obsmat_cg_fname is None:
            #TODO: how to handle this case?
            logger.warning(f"No observation matrix file provided for CG. Using identity matrix instead.")
            obsmat_operator_cg = None
        else:
            logger.debug(f"Loading observation matrix from {obsmat_cg_fname}")
            obsmat_operator_cg = mb.io.load_all_obsmat(
                obsmat_cg_fname, 
                size_obsmat=int(binary_mask.sum()), 
                nstokes=2, 
                kind='precomputations_indices', 
                mask_stacked=None
            )
        
        path_rhs_obsmat = manager.get_path_to_obsmat_rhs()
        if path_rhs_obsmat is None:
            #TODO: how to handle this case?
            logger.warning(f"RHS observation matrix file {path_rhs_obsmat} does not exist. Using identity matrix instead.")
            obsmat_operator_rhs = None
        else:
            logger.debug(f"Loading full-sky transpose observation matrix from {path_rhs_obsmat}")
            obsmat_operator_rhs = mb.io.load_all_obsmat(
                path_rhs_obsmat, 
                size_obsmat=int(binary_mask.sum()), 
                nstokes=2, 
                kind='precomputations_indices', 
                mask_stacked=None,
                return_transpose=config.parametric_sep_pars.return_transpose_rhs
            )

        path_diag_obsmat = manager.get_path_to_diag_obsmat()
        if not Path(path_diag_obsmat).exists():
            logger.warning(f"Diagonal observation matrix file {path_diag_obsmat} does not exist. Using identity matrix instead.")
            diag_obsmat_matrices = None
        else:
            logger.debug(f"Loading diagonal observation matrices from {path_diag_obsmat}")
            diag_obsmat_matrices = np.load(path_diag_obsmat)

    timer = Timer()
    timer.start("do-compsep")

    if config.parametric_sep_pars.passband_int:
        logger.info("Using passband-integration for the component separation step.")
        # config.map_sets = passband.passband_constructor(
        #     config, manager, passband_int=config.parametric_sep_pars.passband_int
        # )
        # passbands_norm = passband.fgbuster_passband(config.map_sets)
        # instrument = {"frequency": passbands_norm}
        raise NotImplementedError(
            "Megabuster does not support passband integration yet. "
            "Please use FGBuster for this feature."
        )
    else:
        instrument = {"frequency": config.frequencies}
    if config.parametric_sep_pars.include_synchrotron:
        components = ['cmb', 'dust', 'synchrotron']  # TODO move default to config
    else:
        components = ['cmb', 'dust']  # TODO move default to config

    # get the 'options' through the appropriate method which returns a dict
    options = config.parametric_sep_pars.get_minimize_options_as_dict()
    tol = config.parametric_sep_pars.minimize_tol
    # method = config.parametric_sep_pars.minimize_method

    # FGBuster's weighted component separation used hp.UNSEEN to ignore masked pixels
    # If put to 0, I don't think they weigh on the outcome but it slows the process down and can result in warnings/errors
    
    freq_maps_preprocessed_QU_masked = mask.apply_binary_mask(
        freq_maps_preprocessed[:, 1:], binary_mask, unseen=False
    )
    noisecov_QU_masked = mask.apply_binary_mask(noisecov[:, 1:], binary_mask, unseen=False)
    inverse_noisecov_QU_masked = np.zeros_like(noisecov_QU_masked)
    inverse_noisecov_QU_masked[noisecov_QU_masked != 0] = 1./noisecov_QU_masked[noisecov_QU_masked != 0]

    res = mb.compsep.perform_compsep(
        first_guess_params={'beta_dust': np.array(1.54), 'beta_pl': np.array(-3.0)},
        fixed_params={'temp_dust': 20.0},
        sky_map=freq_maps_preprocessed_QU_masked,
        frequencies=np.array(config.frequencies),
        invN_matrix=inverse_noisecov_QU_masked,
        binary_mask=binary_mask,
        obs_mat_operator=obsmat_operator_cg, 
        obsmat_operator_rhs=obsmat_operator_rhs,
        diag_obsmat_matrices=diag_obsmat_matrices,
        max_iter=options['maxiter'],
        tol=tol,
        ordering_parameter=['beta_dust', 'beta_pl'], 
        ordering_component=components,
    )
    
    logger.info(f"Success: {res.success} -> {res.message}")
    logger.info(f"Spectral parameters {res.params} -> {res.x}")
    timer.stop("do-compsep")

    return res


def save_compsep_results(manager: DataManager, res, id_sim: int | None = None):
    path = manager.get_path_to_components(sub=id_sim)
    path.mkdir(parents=True, exist_ok=True)
    fname_results = manager.get_path_to_compsep_results(sub=id_sim)
    fname_compmaps = manager.get_path_to_components_maps(sub=id_sim)
    res_dict = {}
    for attr in dir(res):
        if (
            not attr.startswith("__") and attr != "s"
        ):  # remove component maps to avoid saving twice.
            if isinstance(getattr(res, attr), callable):
                continue # Skip callable attributes 
            res_dict[attr] = getattr(res, attr)
    # Saving result dict
    logger.info(f"Saving compsep results to {fname_results}")
    np.savez(fname_results, **res_dict)
    # Saving component maps
    logger.info(f"Saving component maps to {fname_compmaps}")
    np.save(fname_compmaps, res.s)


def compsep_and_save(config: Config, manager: DataManager, id_sim: int | None = None):
    with Timer("weighted-compsep"):
        res = weighted_comp_sep(manager, config, id_sim=id_sim)
    save_compsep_results(manager, res, id_sim=id_sim)
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
                logger.info(f"Distributing work to {executor.num_workers} workers")  # pyright: ignore[reportAttributeAccessIssue]
                func = partial(compsep_and_save, config, manager)
                for result in executor.map(func, range(n_sim_sky), unordered=True):  # pyright: ignore[reportAttributeAccessIssue]
                    logger.info(f"Finished component separation on map {result + 1} / {n_sim_sky}")


if __name__ == "__main__":
    main()
