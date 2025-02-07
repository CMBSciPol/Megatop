import argparse
import multiprocessing as mp
from pathlib import Path

import fgbuster as fg
import healpy as hp
import numpy as np
from fgbuster.component_model import CMB, Dust, Synchrotron
from fgbuster.mixingmatrix import MixingMatrix

from megatop import Config, DataManager
from megatop.utils import Timer, logger, mask


def weighted_comp_sep(manager: DataManager, config: Config):
    with Timer("load-covmat"):
        noisecov_fname = manager.path_to_pixel_noisecov
        logger.debug(f"Loading covmat from {noisecov_fname}")
        noisecov = np.load(noisecov_fname)

    with Timer("load-maps"):
        preproc_maps_fname = manager.get_path_to_preprocessed_maps()
        logger.debug(f"Loading input maps from {preproc_maps_fname}")
        freq_maps_preprocessed = np.load(preproc_maps_fname)

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
    A_ev = A.evaluator(np.array(instrument["frequency"]))
    A_maxL = A_ev(res.x)  # pyright: ignore[reportCallIssue]
    res.A_maxL = A_maxL

    W_maxL = np.einsum("ijsp, jf, fsp -> ifsp", res.invAtNA[:, :], A_maxL.T, 1 / noisecov_QU_masked)
    res.W_maxL = W_maxL

    logger.info(f"Success: {res.success} -> {res.message}")
    logger.info(f"Spectral parameters {res.params} -> {res.x}")
    timer.stop("do-compsep")

    return res


def save_compsep_results(manager: DataManager, res):
    manager.path_to_components.mkdir(parents=True, exist_ok=True)
    fname_results = manager.path_to_compsep_results
    fname_compmaps = manager.path_to_components_maps
    res_dict = {}
    for attr in dir(res):
        if (
            not attr.startswith("__") and attr != "s"
        ):  # remove component maps to avoid saving twice.
            res_dict[attr] = getattr(res, attr)
    # Saving result dict
    logger.info(f"Saving compsep results to {fname_results}")
    np.savez(fname_results, **res_dict)
    # Saving component maps
    logger.info(f"Saving component maps to {fname_compmaps}")
    np.save(fname_compmaps, res.s)


def compsep_and_save(args, id_sim=None):
    if id_sim is None:  # Running only one simulation
        if args.config is None:
            logger.warning("No config file provided, using example config")
            config = Config.get_example()
        else:
            config = Config.from_yaml(args.config)
    else:
        if not args.config_root:
            logger.warning("No config root provided, required for multiple simulations. exiting")
            raise AttributeError
        fname_config = args.config_root.with_name(f"{args.config_root.name}_{id_sim:04d}.yml")
        config = Config.from_yaml(fname_config)
    manager = DataManager(config)
    manager.dump_config()
    with Timer("weighted-compsep"):
        res = weighted_comp_sep(manager, config)
    save_compsep_results(manager, res)


def main():
    parser = argparse.ArgumentParser(description="Component separation")
    parser.add_argument("--config", type=Path, help="config file")
    parser.add_argument(
        "--config_root", type=Path, help="config file root (will be appended  by {id_sim:04d})"
    )
    parser.add_argument("--Nsims", type=int, help="Number of simulations performed")
    parser.add_argument(
        "--nomultiproc", action="store_true", help="don't use multprocessing parallelisation"
    )
    args = parser.parse_args()

    if args.config:  # Prioritize --config if provided
        compsep_and_save(args)
        return

    if args.config_root:  # Multiple simulations mode
        if not args.Nsims:
            logger.warning("Nsims not specified, will only run one ")
            Nsims = 1
        else:
            Nsims = args.Nsims

        num_workers = 1 if args.nomultiproc else min(mp.cpu_count(), Nsims)
        logger.info(f"Using {num_workers} worker processes")
        if num_workers > 1:
            mp.set_start_method("spawn", force=True)  # Ensure a clean multiprocessing start
            with mp.Pool(num_workers) as pool:
                pool.starmap(
                    compsep_and_save,
                    [(args, id_sim) for id_sim in range(Nsims)],
                )
        else:
            for id_sim in range(Nsims):
                compsep_and_save(args, id_sim)
    else:
        # Default case: no arguments provided, run single simulation with example config
        compsep_and_save(args)


if __name__ == "__main__":
    main()
