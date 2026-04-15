from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import healpy as hp
import numpy as np
from numpy.typing import NDArray

from megatop import Config, DataManager
from megatop.utils import Timer, function_timer, logger, mask, mock, passband
from megatop.utils.mpi import get_world
from megatop.utils.TF_utils import get_alms_from_cls, power_law_cl

if TYPE_CHECKING:
    from mpi4py.MPI import Comm

_POOL_EXECUTOR_THRESHOLD = 2


@function_timer("get-noise-map")
def get_noise(
    config: Config, binary_mask: NDArray, common_nhits_map: NDArray, id_sim: int = 0
) -> NDArray:
    fsky_nhits = common_nhits_map.mean()
    noise_freq_maps = mock.get_full_sky_noise_freq_maps(
        config.map_sets,
        config.noise_sim_pars,
        fsky_nhits=fsky_nhits,
        nside=config.nside,
        id_sim=id_sim,
    )
    logger.debug(f"Noise maps has shape {noise_freq_maps.shape}")

    if config.noise_sim_pars.include_nhits:
        _ = mock.include_hits_noise(noise_freq_maps, common_nhits_map, binary_mask)

    return noise_freq_maps


@function_timer("get-cmb-map")
def get_cmb(manager: DataManager, config: Config, id_sim: int = 0) -> NDArray:
    # Performing the CMB simulation with synfast
    logger.debug("Computing CMB map from fiducial spectra")

    # incorporate realization id into the seed if CMB is not fixed
    seed = [config.map_sim_pars.cmb_seed]
    if not config.map_sim_pars.single_cmb:
        seed.append(id_sim)
    logger.debug(f"CMB {seed = }")

    Cl_cmb_model = mock.get_Cl_CMB_model_from_manager(manager)
    cmb_map = mock.generate_map_cmb(Cl_cmb_model, config.nside, cmb_seed=seed)
    logger.debug(f"CMB map has shape {cmb_map.shape}")
    return cmb_map


@function_timer("get-fg-map")
def get_foregrounds(config: Config) -> NDArray:
    # Generating pysm foreground simulations
    logger.debug(f"Generating pysm sky {config.sky_model}")
    fg_freq_maps = mock.generate_map_fgs_pysm(
        config.map_sets,
        config.nside,
        config.map_sim_pars.sky_model,
    )
    logger.debug(f"Foreground map has shape {fg_freq_maps.shape}")
    return fg_freq_maps


@function_timer("load-obsmats")
def load_obsmat(manager: DataManager, config: Config):
    logger.info("Loading observation matrices")
    return mock.load_observation_matrix(
        config.nside, config.map_sets, manager.get_obsmat_filenames()
    )


@function_timer("save-simu")
def save_simu(
    manager: DataManager,
    simulated_maps: NDArray,
    id_sim: int | None = None,
    is_noise: bool = False,
) -> None:
    """Save a sky realization."""
    # get appropriate filenames based on type
    filenames = (
        manager.get_noise_maps_filenames(id_sim) if is_noise else manager.get_maps_filenames(id_sim)
    )

    # save the maps
    for i, fname in enumerate(filenames):
        msg = "Saving noise simulation" if is_noise else "Saving simulated sky"
        logger.debug(f"{msg} to {fname}")
        hp.write_map(
            fname,
            simulated_maps[i],
            dtype=["float64", "float64", "float64"],
            overwrite=True,
        )


@function_timer("save-TFsims")
def save_TFsims(
    manager: DataManager,
    unfiltered_maps_TEB: NDArray,
    filtered_maps_TEB: NDArray,
    id_sim: int | None = None,
) -> None:
    """Save an unfiltered and filtered TF realization."""
    # get appropriate filenames based on type
    filenames_unfiltered, filenames_filtered = manager.get_maps_sim_for_TF_filenames(id_sim)

    # save the maps
    for f in range(len(filenames_unfiltered)):  # loop over frequencies
        for j, s in enumerate(["T", "E", "B"]):  # loop over pure T, E, B
            msg = f"Saving unfiltered TF pure_{s} simulation"
            logger.info(f"{msg} to {filenames_unfiltered[f][j]}")
            hp.write_map(
                filenames_unfiltered[f][j],
                unfiltered_maps_TEB[j, f],
                dtype=["float64", "float64", "float64"],
                overwrite=True,
            )
            msg = f"Saving filtered TF pure_{s} simulation"
            logger.info(f"{msg} to {filenames_filtered[f][j]}")
            hp.write_map(
                filenames_filtered[f][j],
                filtered_maps_TEB[j, f],
                dtype=["float64", "float64", "float64"],
                overwrite=True,
            )


def _map(func, iterable, comm: Comm, force_seq: bool = False):
    """Map function over iterable, using MPICommExecutor if available.

    Args:
        func: Function to map over iterable.
        iterable: Iterable to map function over.
        comm: MPI communicator.
        force_seq: Force sequential processing.

    Yields:
        Results of mapping function over iterable.
    """
    if force_seq or comm is None or comm.Get_size() < _POOL_EXECUTOR_THRESHOLD:
        # Process sequentially
        logger.info("Processing sequentially")
        for result in map(func, iterable):
            yield result
    else:
        # Use CommExecutor for parallel processing
        from mpi4py.futures import MPICommExecutor

        with MPICommExecutor(comm=comm) as executor:
            if executor is not None:
                logger.info(f"Distributing work to {executor.num_workers} processes")
                for result in executor.map(func, iterable, unordered=True):
                    yield result


# needs to be defined at the top level for pickling
def func_TF_sims(
    id_sim: int,
    manager: DataManager,
    config: Config,
    binary_mask: NDArray,
    *,
    obsmat_funcs: dict | None = None,
) -> int:
    """Generate pure E and pure B map with power law spectra for Transfer Function Computation."""

    # Getting power law spectra
    logger.debug("Generating power law spectra for TF simulations")
    ell = np.arange(3 * config.nside + 500)
    pl_spectra = power_law_cl(
        ell,
        config.map_sim_pars.TF_power_law_amp,
        config.map_sim_pars.TF_power_law_delta_ell,
        config.map_sim_pars.TF_power_law_index,
    )
    pl_spectra_new_orderhp = np.array(
        [
            pl_spectra["TT"],
            pl_spectra["EE"],
            pl_spectra["BB"],
            pl_spectra["TE"],
            pl_spectra["EB"],
            pl_spectra["TB"],
        ]
    )

    # Generating alms from power law spectra
    alms_TEB = get_alms_from_cls(
        pl_spectra_new_orderhp, 3 * config.nside, seed=config.map_sim_pars.cmb_seed
    )

    logger.info(
        f"Sum of difference between TT and EE alms / len(alms): {np.sum(alms_TEB[0] - alms_TEB[1]) / len(alms_TEB[0])}"
    )
    logger.info(f"alms T = {alms_TEB[0]}")
    logger.info(f"alms E = {alms_TEB[1]}")
    logger.info(f"alms B = {alms_TEB[2]}")

    # Generating pure T, E and B maps from alms:
    map_pure_T = hp.alm2map(alms_TEB * np.array([1, 0, 0])[:, None], nside=config.nside)
    map_pure_E = hp.alm2map(alms_TEB * np.array([0, 1, 0])[:, None], nside=config.nside)
    map_pure_B = hp.alm2map(alms_TEB * np.array([0, 0, 1])[:, None], nside=config.nside)

    unfiltered_freq_map_pure_T = np.array([map_pure_T] * len(config.frequencies))
    unfiltered_freq_map_pure_E = np.array([map_pure_E] * len(config.frequencies))
    unfiltered_freq_map_pure_B = np.array([map_pure_B] * len(config.frequencies))

    # apply filtering
    if obsmat_funcs is not None:
        with Timer("filter-pure-T-E-B-maps"):
            filtered_freq_map_pure_T = unfiltered_freq_map_pure_T.copy()
            filtered_freq_map_pure_E = unfiltered_freq_map_pure_E.copy()
            filtered_freq_map_pure_B = unfiltered_freq_map_pure_B.copy()

            for i_f, (key, func) in enumerate(obsmat_funcs.items()):
                logger.debug(f"Filtering {key} channel")
                filtered_freq_map_pure_T[i_f] = mock.apply_observation_matrix(
                    func, filtered_freq_map_pure_T[i_f]
                )
                filtered_freq_map_pure_E[i_f] = mock.apply_observation_matrix(
                    func, filtered_freq_map_pure_E[i_f]
                )
                filtered_freq_map_pure_B[i_f] = mock.apply_observation_matrix(
                    func, filtered_freq_map_pure_B[i_f]
                )
    else:
        msg_no_obsmat = (
            "No observation matrices provided for filtering. Please provide obsmat_funcs."
        )
        raise ValueError(msg_no_obsmat)

    # mask unobserved pixels
    _ = mask.apply_binary_mask(unfiltered_freq_map_pure_T, binary_mask, unseen=False)
    _ = mask.apply_binary_mask(unfiltered_freq_map_pure_E, binary_mask, unseen=False)
    _ = mask.apply_binary_mask(unfiltered_freq_map_pure_B, binary_mask, unseen=False)
    _ = mask.apply_binary_mask(filtered_freq_map_pure_T, binary_mask, unseen=False)
    _ = mask.apply_binary_mask(filtered_freq_map_pure_E, binary_mask, unseen=False)
    _ = mask.apply_binary_mask(filtered_freq_map_pure_B, binary_mask, unseen=False)

    # save results
    save_TFsims(
        manager,
        np.array(
            [unfiltered_freq_map_pure_T, unfiltered_freq_map_pure_E, unfiltered_freq_map_pure_B]
        ),
        np.array([filtered_freq_map_pure_T, filtered_freq_map_pure_E, filtered_freq_map_pure_B]),
        id_sim=id_sim,
    )

    return id_sim


# needs to be defined at the top level for pickling
def func_signal(
    id_sim: int,
    manager: DataManager,
    config: Config,
    binary_mask: NDArray,
    common_nhits_map: NDArray,
    *,
    obsmat_funcs: dict | None = None,
) -> int:
    """Generate a sky realization."""
    # construct passbands if necessary
    config.map_sets = passband.passband_constructor(
        config, manager, passband_int=config.map_sim_pars.passband_int
    )
    if config.map_sim_pars.passband_int:
        logger.info("Using passband-integration for the mocker step.")

    # generate the components
    cmb = get_cmb(manager, config, id_sim=id_sim)
    fg = get_foregrounds(config)
    noise = get_noise(config, binary_mask, common_nhits_map, id_sim=id_sim)

    # broadcast CMB to all frequencies
    sky = cmb[None, ...] + fg

    # apply beam and pixel window function correction
    with Timer("beam-freq-maps"):
        for i_f, _f in enumerate(config.frequencies):
            sky[i_f] = mock.beam_winpix_correction(config.nside, sky[i_f], config.beams[i_f])

    # apply filtering
    if obsmat_funcs is not None:
        with Timer("filter-freq-maps"):
            for i_f, (key, func) in enumerate(obsmat_funcs.items()):
                logger.debug(f"Filtering {key} channel")
                sky[i_f] = mock.apply_observation_matrix(func, sky[i_f])

    # add noise
    sky += noise

    # mask unobserved pixels
    _ = mask.apply_binary_mask(sky, binary_mask, unseen=False)

    # save results
    save_simu(manager, sky, id_sim=id_sim, is_noise=False)

    return id_sim


# needs to be defined at the top level for pickling
def func_noise(
    manager: DataManager,
    config: Config,
    binary_mask: NDArray,
    common_nhits_map: NDArray,
    id_sim: int,
) -> int:
    """Generate a noise realization."""
    noise = get_noise(config, binary_mask, common_nhits_map, id_sim=id_sim)
    _ = mask.apply_binary_mask(noise, binary_mask, unseen=False)
    save_simu(manager, noise, id_sim=id_sim, is_noise=True)
    return id_sim


def process_signal(config: Config, manager: DataManager, comm: Comm):
    rank = 0 if comm is None else comm.Get_rank()
    n_sim = config.map_sim_pars.n_sim

    if n_sim == 0:
        return

    if rank == 0:
        logger.info(f"Generating {n_sim} sky realizations")

    # Load necessary data
    binary_mask = hp.read_map(manager.path_to_binary_mask)
    common_nhits_map = hp.read_map(manager.path_to_common_nhits_map)
    func = partial(
        func_signal,
        manager=manager,
        config=config,
        binary_mask=binary_mask,
        common_nhits_map=common_nhits_map,
    )

    if filtering := config.map_sim_pars.filter_sims:
        # Load the obsmat(s) for our map set(s)
        obsmat_funcs = load_obsmat(manager, config)
        func = partial(func, obsmat_funcs=obsmat_funcs)

    for result in _map(func, range(n_sim), comm, force_seq=filtering):
        logger.info(f"Finished sky realization {result + 1} / {n_sim}")


def process_noise(config: Config, manager: DataManager, comm: Comm):
    rank = 0 if comm is None else comm.Get_rank()
    n_sim = config.noise_sim_pars.n_sim

    if n_sim == 0:
        return

    if rank == 0:
        logger.info(f"Generating {n_sim} noise realizations")

    # Load necessary data
    binary_mask = hp.read_map(manager.path_to_binary_mask)
    common_nhits_map = hp.read_map(manager.path_to_common_nhits_map)
    func = partial(func_noise, manager, config, binary_mask, common_nhits_map)

    for result in _map(func, range(n_sim), comm):
        logger.info(f"Finished noise realization {result + 1} / {n_sim}")


def process_TF_sims(config: Config, manager: DataManager, comm: Comm):
    """Generate pure T, E and pure B map with power law spectra for Transfer Function Computation."""
    rank = 0 if comm is None else comm.Get_rank()
    n_sim = config.map_sim_pars.TF_n_sim

    if n_sim == 0:
        return

    if rank == 0:
        logger.info(f"Generating {n_sim} TF simulations")

    # Load necessary data
    binary_mask = hp.read_map(manager.path_to_binary_mask)

    func = partial(
        func_TF_sims,
        manager=manager,
        config=config,
        binary_mask=binary_mask,
    )

    if filtering := config.map_sim_pars.filter_sims:
        # Load the obsmat(s) for our map set(s)
        obsmat_funcs = load_obsmat(manager, config)
        func = partial(func, obsmat_funcs=obsmat_funcs)

    for result in _map(func, range(n_sim), comm, force_seq=filtering):
        logger.info(f"Finished TF simulation {result + 1} / {n_sim}")


def _load_masks(manager: DataManager, config: Config):
    binary_mask = hp.read_map(manager.path_to_binary_mask)
    list_hitmapname = [manager.path_to_nhits_map(m) for m in config.map_sets]
    nhits_maps = mask.read_nhits_maps(list_hitmapname, nside=config.nside)
    return binary_mask, nhits_maps


def main_signal():
    """Entry point for generating a single sky realization."""
    parser = argparse.ArgumentParser(description="Generate a single sky realization")
    parser.add_argument("--config", type=Path, required=True, help="config file")
    parser.add_argument("--sim", type=int, required=True, help="simulation index to generate")

    args = parser.parse_args()
    config = Config.load_yaml(args.config)
    manager = DataManager(config)
    manager.create_output_dirs(config.map_sim_pars.n_sim, config.noise_sim_pars.n_sim)

    binary_mask, nhits_maps = _load_masks(manager, config)
    func_signal(args.sim, manager, config, binary_mask, nhits_maps)


def main_noise():
    """Entry point for generating a single noise realization."""
    parser = argparse.ArgumentParser(description="Generate a single noise realization")
    parser.add_argument("--config", type=Path, required=True, help="config file")
    parser.add_argument("--sim", type=int, required=True, help="simulation index to generate")

    args = parser.parse_args()
    config = Config.load_yaml(args.config)
    manager = DataManager(config)
    manager.create_output_dirs(config.map_sim_pars.n_sim, config.noise_sim_pars.n_sim)

    binary_mask, nhits_maps = _load_masks(manager, config)
    func_noise(manager, config, binary_mask, nhits_maps, args.sim)


def main():
    parser = argparse.ArgumentParser(
        description="Script for generating signal and noise realizations"
    )
    parser.add_argument("--config", type=Path, required=True, help="config file")

    # Parse arguments
    args = parser.parse_args()
    config = Config.load_yaml(args.config)
    manager = DataManager(config)

    # Split the world communicator based on map sets
    world, rank, size = get_world()
    num_sets = len(config.map_sets)
    color = rank % num_sets
    if world is not None:
        scomm = world.Split(color=color, key=rank)
        srank = scomm.Get_rank()
        ssize = scomm.Get_size()
        num_groups = min(size, num_sets)
    else:
        scomm = None
        srank = ssize = 0
        num_groups = 1

    if rank == 0:
        manager.dump_config()
        manager.create_output_dirs(config.map_sim_pars.n_sim, config.noise_sim_pars.n_sim)
    if world is not None:
        world.Barrier()

    # Split the configuration
    sconf = config.split_map_sets(num_groups, color=color)

    if srank == 0:
        num_sets = len(sconf.map_sets)
        sets = ", ".join(s.name for s in sconf.map_sets)
        logger.info(f"Group {color} (size {ssize}) responsible for {num_sets} map sets ({sets})")

    # Update the manager's configuration before processing
    manager = DataManager(sconf)
    process_signal(sconf, manager, scomm)
    process_noise(sconf, manager, scomm)

    if config.map_sim_pars.generate_sims_for_TF:
        process_TF_sims(sconf, manager, scomm)


if __name__ == "__main__":
    main()
