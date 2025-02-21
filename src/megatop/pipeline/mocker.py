import argparse
from functools import partial
from pathlib import Path

import healpy as hp
import numpy as np
from mpi4py.futures import MPICommExecutor
from numpy.typing import NDArray

from megatop import Config, DataManager
from megatop.utils import Timer, logger, mask, mock
from megatop.utils.mpi import get_world


def generate_simu(
    manager: DataManager,
    config: Config,
    obsmat_funcs: dict | None = None,
    components: str | list[str] = "all",
):
    """Generate a single realization of the sky maps."""
    timer = Timer()

    timer.start("simulate-one-sky-map")
    binary_mask = hp.read_map(manager.path_to_binary_mask)  # TODO read_binary in manager ?
    fsky_binary = sum(binary_mask) / len(binary_mask)

    if components == "all":
        components = ["cmb", "fg", "noise"]

    if "noise" in components:
        # Creating noise maps
        timer.start("compute-noise-maps")

        if config.noise_sim_pars.noise_option == "white_noise":
            logger.info("Simulation has white noise only")
            # TODO: refactor to use config (requires changes in utils/mock.py)
            _, map_white_noise_levels = mock.get_noise(config, fsky_binary)
            noise_freq_maps = mock.get_noise_map_from_white_noise(
                config.frequencies, config.nside, map_white_noise_levels
            )
            logger.debug(f"Noise maps has shape {noise_freq_maps.shape}")

        elif config.noise_sim_pars.noise_option == "no_noise":
            logger.info("Simulation has NO NOISE")
            noise_freq_maps = None

        elif config.noise_sim_pars.noise_option == "noise_spectra":
            logger.info("Simulation has noise from full spectra")
            n_ell, _ = mock.get_noise(config, fsky_binary)
            noise_freq_maps = mock.get_noise_map_from_noise_spectra(
                config.frequencies, config.nside, n_ell
            )
            logger.debug(f"Noise maps has shape {noise_freq_maps.shape}")

        timer.stop("compute-noise-maps")
    else:
        noise_freq_maps = None
    if noise_freq_maps is not None and config.noise_sim_pars.include_nhits:
        noise_freq_maps = mock.include_hits_noise(
            noise_freq_maps, hp.read_map(manager.path_to_nhits_map), binary_mask
        )  # TODO move reading of maps to manager

    if "cmb" in components:
        # Performing the CMB simulation with synfast
        timer.start("compute-cmb-map")
        logger.info("Computing CMB map from fiducial spectra")

        Cl_cmb_model = mock.get_Cl_CMB_model_from_manager(manager)
        cmb_map = mock.generate_map_cmb(
            Cl_cmb_model, config.nside, fixed_cmb_seed=config.map_sim_pars.fixed_cmb_seed
        )

        logger.debug(f"CMB map has shape {cmb_map.shape}")
        timer.stop("compute-cmb-map")
    else:
        cmb_map = np.zeros((3, hp.nside2npix(config.nside)))

    if "fg" in components:
        # Generating pysm foreground simulations
        timer.start("generate-foregrounds-map")
        logger.info(f"Generating pysm sky {config.sky_model}")
        fg_freq_maps = mock.generate_map_fgs_pysm(
            config.frequencies, config.nside, config.map_sim_pars.sky_model
        )

        logger.debug(f"Foreground map has shape {fg_freq_maps.shape}")
        timer.stop("generate-foregrounds-map")
    else:
        fg_freq_maps = np.zeros((len(config.frequencies), 3, hp.nside2npix(config.nside)))

    combined_freq_maps = np.zeros((len(config.frequencies), 3, hp.nside2npix(config.nside)))
    combined_freq_maps_beamed = np.zeros((len(config.frequencies), 3, hp.nside2npix(config.nside)))

    if components == ["noise"]:
        noise_freq_maps[..., np.where(binary_mask == 0)[0]] = 0
        logger.info("Only noise maps generated")
        timer.stop("simulate-one-sky-map")
        return noise_freq_maps, None, None

    with Timer("beam-freq-maps"):
        for i_f, _f in enumerate(config.frequencies):
            combined_freq_maps[i_f] = cmb_map + fg_freq_maps[i_f]
            combined_freq_maps_beamed[i_f] = mock.beam_winpix_correction(
                config.nside, cmb_map + fg_freq_maps[i_f], config.beams[i_f]
            )

    if obsmat_funcs is not None:
        with Timer("filter-freq-maps"):
            for i_f, map_set_name in enumerate(obsmat_funcs.keys()):
                logger.info(f"Filtering {config.frequencies[i_f]} channel")
                combined_freq_maps_beamed[i_f] = mock.apply_observation_matrix(
                    obsmat_funcs[map_set_name], combined_freq_maps_beamed[i_f]
                )
    if noise_freq_maps is not None:
        for i_f in range(len(config.frequencies)):
            combined_freq_maps[i_f] += noise_freq_maps[i_f]
            combined_freq_maps_beamed[i_f] += noise_freq_maps[i_f]

    combined_freq_maps = mask.apply_binary_mask(combined_freq_maps, binary_mask, unseen=False)
    combined_freq_maps_beamed = mask.apply_binary_mask(
        combined_freq_maps_beamed, binary_mask, unseen=False
    )

    timer.stop("simulate-one-sky-map")

    return noise_freq_maps, combined_freq_maps, combined_freq_maps_beamed


def load_obsmat(manager: DataManager, config: Config):
    # TODO: move to data manager
    logger.info("Loading observation matrices")
    with Timer(thread="load-obsmat"):
        return mock.load_observation_matrix(
            config.nside, config.map_sets, manager.get_obsmat_filenames()
        )


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


def process_simu(
    config: Config,
    manager: DataManager,
    id_sim: int | None = None,
    sim_signal: bool = True,  # True: simulate signal (with noise) ; False: simulate noise only
    obsmat_funcs: dict | None = None,
) -> None | int:
    """Generate and save a single realization of the sky maps."""
    if sim_signal:
        noise_freq_maps, _, combined_freq_maps_beamed = generate_simu(
            manager, config, obsmat_funcs=obsmat_funcs
        )
        save_simu(manager, combined_freq_maps_beamed, id_sim=id_sim, is_noise=False)
    else:
        noise_freq_maps, _, _ = generate_simu(manager, config, components=["noise"])
        save_simu(manager, noise_freq_maps, id_sim=id_sim, is_noise=True)
    return id_sim


def main():
    world, rank, size = get_world()

    parser = argparse.ArgumentParser(
        description="Script for generating signal and noise realizations",
        epilog="mpi4py is required to run this script",
    )
    parser.add_argument("--config", type=Path, required=True, help="config file")

    # Parse arguments
    args = parser.parse_args()
    config = Config.from_yaml(args.config)
    manager = DataManager(config)
    if rank == 0:
        manager.dump_config()

    # Signal simulations:
    n_sim_sky = config.map_sim_pars.n_sim
    if n_sim_sky == 0:
        logger.info("No sky realizations generated")
    else:
        if rank == 0:
            logger.info(f"Generating {n_sim_sky} sky realizations")
        if config.map_sim_pars.filter_sims:
            # We need to load the obsmat(s)
            # FIXME: use parallel loading
            if rank == 0:
                msg = "Parallelization of obsmats not implemented yet, ignoring other processes"
                logger.warning(msg)
                obsmat_funcs = load_obsmat(manager, config)
                for id_sim in range(n_sim_sky):
                    process_simu(
                        config, manager, id_sim=id_sim, sim_signal=True, obsmat_funcs=obsmat_funcs
                    )
        else:
            with MPICommExecutor() as executor:
                if executor is not None:
                    logger.info(f"Distributing work to {executor.num_workers} workers")  # pyright: ignore[reportAttributeAccessIssue]
                    func = partial(process_simu, config, manager, sim_signal=True)
                    for result in executor.map(func, range(n_sim_sky), unordered=True):  # pyright: ignore[reportAttributeAccessIssue]
                        logger.info(f"Finished sky realization {result + 1} / {n_sim_sky}")

    # Noise simulations:
    if (
        config.noise_sim_pars.noise_option == "no_noise"
    ):  # If noise_option == no_noise, no noise simulations are generated, no matter noise_sim_pars.n_sims
        logger.info(
            f" Parameter noise_option is set to {config.noise_sim_pars.noise_option}: no noise realizations generated"
        )
        return
    n_sim_noise = config.noise_sim_pars.n_sim
    with MPICommExecutor() as executor:
        if executor is not None:
            logger.info(f"Generating {n_sim_noise} noise realizations")
            logger.info(f"Distributing work to {executor.num_workers} workers")  # pyright: ignore[reportAttributeAccessIssue]
            func = partial(process_simu, config, manager, sim_signal=False)
            for result in executor.map(func, range(n_sim_noise), unordered=True):  # pyright: ignore[reportAttributeAccessIssue]
                logger.info(f"Finished noise realization {result + 1} / {n_sim_noise}")
    return


if __name__ == "__main__":
    main()
