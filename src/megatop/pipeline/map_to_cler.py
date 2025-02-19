import argparse
import multiprocessing as mp
from pathlib import Path

import healpy as hp
import numpy as np
import pymaster as nmt

from megatop import Config, DataManager
from megatop.utils import Timer, logger, mask
from megatop.utils.spectra import (
    compute_auto_cross_cl_from_maps_list,
    create_binning,
    get_common_beam_wpix,
    initialize_nmt_workspace,
    limit_namaster_output,
)


def spectra_estimation(manager: DataManager, config: Config):
    with Timer("load-component-maps"):
        comp_maps = np.load(manager.path_to_components_maps)

    # Creating/loading bins
    bin_low, bin_high, bin_centre = create_binning(
        config.nside, config.map2cl_pars.delta_ell, end_first_bin=config.lmin
    )

    bin_index_lminlmax = np.where((bin_low >= config.lmin) & (bin_high <= config.lmax))[0]

    manager.path_to_binning.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        manager.path_to_binning,
        bin_low=bin_low,
        bin_high=bin_high,
        bin_centre=bin_centre,
        bin_index_lminlmax=bin_index_lminlmax,
        bin_centre_lminlmax=bin_centre[bin_index_lminlmax],
    )
    nmt_bins = nmt.NmtBin.from_edges(bin_low, bin_high + 1)

    # Loading analysis mask
    mask_analysis = hp.read_map(manager.path_to_analysis_mask)
    binary_mask = hp.read_map(manager.path_to_binary_mask).astype(bool)

    # Generating effective beam
    # TODO: If input maps are used instead of preprocessed ones, the effective beam after compsep must be computed.

    effective_beam_CMB = get_common_beam_wpix(
        config.pre_proc_pars.common_beam_correction, config.nside
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

    # Testing the function

    with Timer("estimate-spectra"):
        comp_maps = mask.apply_binary_mask(comp_maps, binary_mask)

        comp_dict = {"CMB": comp_maps[0], "Dust": comp_maps[1], "Synch": comp_maps[2]}
        # TODO: when components will be added in .yml for the comp-sep steps the keys of the dictionary should adapt to that

        all_Cls = compute_auto_cross_cl_from_maps_list(
            comp_dict,
            mask_analysis,
            effective_beam_CMB[:-1],
            workspaceff,
            purify_e=config.map2cl_pars.purify_e,
            purify_b=config.map2cl_pars.purify_b,
            n_iter=config.map2cl_pars.n_iter_namaster,
        )

        # Limiting the output to the desired l range
    return limit_namaster_output(all_Cls, bin_index_lminlmax)


def save_spectra(manager: DataManager, all_Cls: dict):
    manager.path_to_cross_components_spectra.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving estimated spectra to {manager.path_to_cross_components_spectra}")
    np.savez(manager.path_to_cross_components_spectra, **all_Cls)


def map2cl_and_save(args, id_sim=None):
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
        fname_config = args.config_root.with_name(f"{args.config_root.name}_{id_sim:04d}.yaml")
        config = Config.from_yaml(fname_config)
    manager = DataManager(config)
    manager.dump_config()
    with Timer("spectra-estimation"):
        all_Cls = spectra_estimation(manager, config)
    save_spectra(manager, all_Cls=all_Cls)


def main():
    parser = argparse.ArgumentParser(description="Map to CLs")
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
        map2cl_and_save(args)
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
                    map2cl_and_save,
                    [(args, id_sim) for id_sim in range(Nsims)],
                )
        else:
            for id_sim in range(Nsims):
                map2cl_and_save(args, id_sim)
    else:
        # Default case: no arguments provided, run single simulation with example config
        map2cl_and_save(args)


if __name__ == "__main__":
    main()
