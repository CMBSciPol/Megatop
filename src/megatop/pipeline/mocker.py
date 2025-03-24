import argparse
from functools import partial
from pathlib import Path

import healpy as hp
import numpy as np
from mpi4py.futures import MPICommExecutor
from mpi4py.MPI import Comm
from numpy.typing import NDArray

from megatop import Config, DataManager
from megatop.config import NoiseOption
from megatop.utils import Timer, function_timer, logger, mask, mock
from megatop.utils.mpi import get_world

_POOL_EXECUTOR_THRESHOLD = 2


@function_timer("get-noise-map")
def get_noise(config: Config, binary_mask: NDArray, nhits_map: NDArray) -> NDArray:
    noise_option = config.noise_sim_pars.noise_option

    if noise_option == NoiseOption.NOISELESS:
        logger.debug("Simulation has NO NOISE")
        return np.array(0)

    fsky_binary = binary_mask.mean()

    if config.noise_sim_pars.noise_option == NoiseOption.WHITE:
        logger.debug("Simulation has white noise only")
        _, noise_levels = mock.get_noise(config, fsky_binary)
        noise_freq_maps = mock.get_noise_map_from_white_noise(
            config.frequencies, config.nside, noise_levels
        )
        logger.debug(f"Noise maps has shape {noise_freq_maps.shape}")

    elif config.noise_sim_pars.noise_option == NoiseOption.ONE_OVER_F:
        logger.debug("Simulation has noise from full spectra")
        n_ell, _ = mock.get_noise(config, fsky_binary)
        noise_freq_maps = mock.get_noise_map_from_noise_spectra(
            config.frequencies, config.nside, n_ell
        )
        logger.debug(f"Noise maps has shape {noise_freq_maps.shape}")

    if config.noise_sim_pars.include_nhits:
        _ = mock.include_hits_noise(noise_freq_maps, nhits_map, binary_mask)

    return noise_freq_maps


@function_timer("get-cmb-map")
def get_cmb(manager: DataManager, config: Config) -> NDArray:
    # Performing the CMB simulation with synfast
    logger.debug("Computing CMB map from fiducial spectra")
    Cl_cmb_model = mock.get_Cl_CMB_model_from_manager(manager)
    cmb_map = mock.generate_map_cmb(
        Cl_cmb_model, config.nside, cmb_seed=config.map_sim_pars.cmb_seed
    )
    logger.debug(f"CMB map has shape {cmb_map.shape}")
    return cmb_map


@function_timer("get-fg-map")
def get_foregrounds(config: Config) -> NDArray:
    # Generating pysm foreground simulations
    logger.debug(f"Generating pysm sky {config.sky_model}")
    fg_freq_maps = mock.generate_map_fgs_pysm(
        config.frequencies, config.nside, config.map_sim_pars.sky_model
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
    # create the subdirectory for this realization
    if id_sim is not None:
        if is_noise:
            path = manager.get_path_to_noise_maps_sub(id_sim)
        else:
            path = manager.get_path_to_maps_sub(id_sim)
        path.mkdir(parents=True, exist_ok=True)

    # get appropriate filenames based on type
    filenames = (
        manager.get_noise_maps_filenames(sub=id_sim)
        if is_noise
        else manager.get_maps_filenames(sub=id_sim)
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
    if force_seq or comm.Get_size() < _POOL_EXECUTOR_THRESHOLD:
        # Process sequentially
        logger.info("Processing sequentially")
        for result in map(func, iterable):
            yield result

    # Use CommExecutor for parallel processing
    with MPICommExecutor(comm=comm) as executor:  # pyright: ignore[reportArgumentType]
        if executor is not None:
            logger.info(f"Distributing work to {executor.num_workers} processes")  # pyright: ignore[reportAttributeAccessIssue]
            for result in executor.map(func, iterable, unordered=True):  # pyright: ignore[reportAttributeAccessIssue]
                yield result


# needs to be defined at the top level for pickling
def func_signal(
    id_sim: int,
    manager: DataManager,
    config: Config,
    binary_mask: NDArray,
    nhits_map: NDArray,
    *,
    obsmat_funcs: dict | None = None,
) -> int:
    """Generate a sky realization."""

    # incorporate realization id into the seed if CMB is not fixed
    if not config.map_sim_pars.single_cmb:
        config.map_sim_pars.cmb_seed += id_sim  # pyright: ignore[reportOperatorIssue]

    # generate the components
    cmb = get_cmb(manager, config)
    fg = get_foregrounds(config)
    noise = get_noise(config, binary_mask, nhits_map)

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
    nhits_map: NDArray,
    id_sim: int,
) -> int:
    """Generate a noise realization."""
    noise = get_noise(config, binary_mask, nhits_map)
    _ = mask.apply_binary_mask(noise, binary_mask, unseen=False)
    save_simu(manager, noise, id_sim=id_sim, is_noise=True)
    return id_sim


def process_signal(config: Config, manager: DataManager, comm: Comm):
    rank = comm.Get_rank()
    n_sim = config.map_sim_pars.n_sim

    if n_sim == 0:
        return

    if rank == 0:
        logger.info(f"Generating {n_sim} sky realizations")

    # Load necessary data
    binary_mask = hp.read_map(manager.path_to_binary_mask)
    nhits_map = hp.read_map(manager.path_to_nhits_map)
    func = partial(
        func_signal,
        manager=manager,
        config=config,
        binary_mask=binary_mask,
        nhits_map=nhits_map,
    )

    if filtering := config.map_sim_pars.filter_sims:
        # Load the obsmat(s) for out map set(s)
        obsmat_funcs = load_obsmat(manager, config)
        func = partial(func, obsmat_funcs=obsmat_funcs)

    for result in _map(func, range(n_sim), comm, force_seq=filtering):
        logger.info(f"Finished sky realization {result + 1} / {n_sim}")


def process_noise(config: Config, manager: DataManager, comm: Comm):
    rank = comm.Get_rank()
    n_sim = config.noise_sim_pars.n_sim

    if n_sim == 0:
        return

    noise_option = config.noise_sim_pars.noise_option
    if noise_option == NoiseOption.NOISELESS:
        # In noiseless mode, do not simulate noise at all
        return

    if rank == 0:
        logger.info(f"Generating {n_sim} noise realizations")

    # Load necessary data
    binary_mask = hp.read_map(manager.path_to_binary_mask)
    nhits_map = hp.read_map(manager.path_to_nhits_map)
    func = partial(func_noise, manager, config, binary_mask, nhits_map)

    for result in _map(func, range(n_sim), comm):
        logger.info(f"Finished sky realization {result + 1} / {n_sim}")


def main():
    parser = argparse.ArgumentParser(
        description="Script for generating signal and noise realizations",
        epilog="mpi4py is required to run this script",
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
    scomm = world.Split(color=color, key=rank)
    srank = scomm.Get_rank()
    ssize = scomm.Get_size()

    # Now split the configuration for the different groups
    num_groups = min(size, num_sets)

    # We need to handle the CMB seed carefully
    # If not provided, generate a common one that will be shared by all groups
    cmb_seed = config.map_sim_pars.cmb_seed
    if cmb_seed is None:
        if rank == 0 and num_groups > 1:
            # Process 0 generates the seed for everyone from a random source
            rng = np.random.default_rng()
            cmb_seed = rng.integers(2**32)
            logger.debug(f"Common CMB seed: {cmb_seed}")
        config.map_sim_pars.cmb_seed = int(world.bcast(cmb_seed, root=0))

    # Dump the full configuration including the generated seed, before splitting the map sets
    if rank == 0:
        manager.dump_config()
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


if __name__ == "__main__":
    main()
