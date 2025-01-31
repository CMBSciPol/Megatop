import argparse
from pathlib import Path

import healpy as hp
import numpy as np
import pymaster as nmt

from megatop import Config, DataManager
from megatop.utils import Timer, logger, utils
from megatop.utils.spectra import (
    compute_auto_cross_cl_from_maps_list,
    get_common_beam_wpix,
    initialize_nmt_workspace,
)


def spectra_estimation(manager: DataManager, config: Config) -> None:
    timer = Timer()
    timer.start("loading_comp_maps")
    comp_maps = np.load(manager.path_to_components_maps)
    timer.stop("loading_comp_maps", "Loading component maps")

    # Creating/loading bins
    bin_low, bin_high, bin_centre = utils.create_binning(
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
    )
    nmt_bins = nmt.NmtBin.from_edges(bin_low, bin_high + 1)

    # Loading analysis mask
    mask_analysis = hp.read_map(manager.path_to_analysis_mask)
    binary_mask = hp.read_map(manager.path_to_binary_mask).astype(bool)

    # Generating effective beam
    # TODO: If input maps are used instead of preprocessed ones, the effective beam after compsep must be computed.

    effective_beam = get_common_beam_wpix(config.pre_proc_pars.common_beam_correction, config.nside)

    # Initializin workspace
    timer.start("initializing_workspace")

    workspaceff = initialize_nmt_workspace(
        nmt_bins,
        manager.path_to_lensed_scalar,
        config.nside,
        mask_analysis,
        effective_beam[:-1],
        config.map2cl_pars.purify_e,
        config.map2cl_pars.purify_b,
        config.map2cl_pars.n_iter_namaster,
    )

    timer.stop("initializing_workspace", "Initializing workspace")

    # Testing the function
    timer.start("spectra_estimation")

    # if hp.UNSEEN is used in comp-sep, the comp-maps will use it as well which will be a problem for namaster, we regularize it here
    comp_maps *= binary_mask

    comp_dict = {"CMB": comp_maps[0], "Dust": comp_maps[1], "Synch": comp_maps[2]}
    # TODO: when components will be added in .yml for the comp-sep steps the keys of the dictionary should adapt to that

    all_Cls = compute_auto_cross_cl_from_maps_list(
        comp_dict,
        mask_analysis,
        effective_beam,
        workspaceff,
        purify_e=config.map2cl_pars.purify_e,
        purify_b=config.map2cl_pars.purify_b,
        n_iter=config.map2cl_pars.n_iter_namaster,
    )

    manager.path_to_cross_components_spectra.parent.mkdir(parents=True, exist_ok=True)
    np.savez(manager.path_to_cross_components_spectra, **all_Cls)

    timer.stop("spectra_estimation", "Spectra estimation")


def main():
    parser = argparse.ArgumentParser(description="Map to CLs")
    parser.add_argument("--config", type=Path, help="config file")
    args = parser.parse_args()
    if args.config is None:
        logger.warning("No config file provided, using example config")
        config = Config.get_example()
    else:
        config = Config.from_yaml(args.config)
    manager = DataManager(config)
    manager.dump_config()
    spectra_estimation(manager, config)


if __name__ == "__main__":
    main()
