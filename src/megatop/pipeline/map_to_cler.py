import argparse
from functools import partial
from pathlib import Path

import healpy as hp
import numpy as np
import pymaster as nmt
from mpi4py.futures import MPICommExecutor

from megatop import Config, DataManager
from megatop.utils import Timer, logger, mask
from megatop.utils.spectra import (
    compute_auto_cross_cl_from_maps_list,
    create_binning,
    get_common_beam_wpix,
    get_native_post_compsep_beam_wpix,
    initialize_nmt_workspace,
    limit_namaster_output,
)


def spectra_estimation(manager: DataManager, config: Config, id_sim: int | None = None):
    with Timer("load-component-maps"):
        comp_path = manager.get_path_to_components_maps(sub=id_sim)
        print(comp_path)
        comp_maps = np.load(manager.get_path_to_components_maps(sub=id_sim))

    # Creating/loading bins
    bin_low, bin_high, bin_centre = create_binning(
        config.nside, config.map2cl_pars.delta_ell, end_first_bin=config.lmin
    )

    bin_index_lminlmax = np.where((bin_low >= config.lmin) & (bin_high <= config.lmax))[0]

    path = manager.get_path_to_spectra_binning(sub=id_sim)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
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

    if config.parametric_sep_pars.use_native_resolution:
        logger.info(
            "Using native resolution maps. Getting effective beam from component separation results."
        )

        # Loading component separation operator
        A_maxL = np.load(manager.get_path_to_compsep_results(sub=id_sim), allow_pickle=True)[
            "A_maxL"
        ]

        effective_beam_P, effective_beam_T = get_native_post_compsep_beam_wpix(
            config.beams, A_maxL, config.nside
        )

        # Getting only CMB component [0,0] and normalizing by the max of the CMB Temperature effective beam.
        # This should be more consistent with how beams are handled in healpy (TODO:to be checked)
        effective_beam_CMB = effective_beam_P[0, 0] / np.max(effective_beam_T[0, 0])

    else:
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


def save_spectra(manager: DataManager, all_Cls: dict, id_sim: int | None = None):
    path = manager.get_path_to_spectra(sub=id_sim)
    path.mkdir(parents=True, exist_ok=True)
    fname = manager.get_path_to_spectra_cross_components(sub=id_sim)
    logger.info(f"Saving estimated spectra to {fname}")
    np.savez(fname, **all_Cls)


def map2cl_and_save(config: Config, manager: DataManager, id_sim: int | None = None):
    with Timer("spectra-estimation"):
        all_Cls = spectra_estimation(manager, config, id_sim=id_sim)
    save_spectra(manager, all_Cls=all_Cls, id_sim=id_sim)
    return id_sim


def main():
    parser = argparse.ArgumentParser(description="Map to CLs")
    parser.add_argument("--config", type=Path, required=True, help="config file")

    args = parser.parse_args()
    config = Config.load_yaml(args.config)
    manager = DataManager(config)

    n_sim_sky = config.map_sim_pars.n_sim
    if n_sim_sky == 0:
        map2cl_and_save(config, manager, id_sim=None)
    else:
        with MPICommExecutor() as executor:
            if executor is not None:
                logger.info(f"Distributing work to {executor.num_workers} workers")  # pyright: ignore[reportAttributeAccessIssue]
                func = partial(map2cl_and_save, config, manager)
                for result in executor.map(func, range(n_sim_sky), unordered=True):  # pyright: ignore[reportAttributeAccessIssue]
                    logger.info(f"Finished Cl estimation on map {result + 1} / {n_sim_sky}")


if __name__ == "__main__":
    main()
