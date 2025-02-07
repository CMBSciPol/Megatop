import argparse
import multiprocessing as mp
from pathlib import Path

import healpy as hp
import numpy as np

from megatop import Config, DataManager
from megatop.utils import Timer, logger, mask, mock


def make_sims(
    manager: DataManager,
    config: Config,
    obsmats_loaded: bool = False,
    dict_obsmats_func: dict | None = None,
    components: str | list[str] = "all",
):
    timer = Timer()

    # create the directory for the maps
    manager.path_to_maps.mkdir(parents=True, exist_ok=True)

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

        elif config.noise_sim_pars.noise_option == "no_noise":
            logger.info("Simulation has NO NOISE")
            noise_freq_maps = np.zeros((len(config.frequencies), 3, hp.nside2npix(config.nside)))

        elif config.noise_sim_pars.noise_option == "noise_spectra":
            logger.info("Simulation has noise from full spectra")
            n_ell, _ = mock.get_noise(config, fsky_binary)
            noise_freq_maps = mock.get_noise_map_from_noise_spectra(
                config.frequencies, config.nside, n_ell
            )

        logger.debug(f"Noise maps has shape {noise_freq_maps.shape}")
        timer.stop("compute-noise-maps")
    else:
        noise_freq_maps = np.zeros((len(config.frequencies), 3, hp.nside2npix(config.nside)))
    if config.noise_sim_pars.include_nhits:
        noise_freq_maps = mock.include_hits_noise(
            noise_freq_maps, hp.read_map(manager.path_to_nhits_map), binary_mask
        )  # TODO move reading of maps to manager

    if "cmb" in components:
        # Performing the CMB simulation with synfast
        timer.start("compute-cmb-map")
        logger.info("Computing CMB map from fiducial spectra")

        Cl_cmb_model = mock.get_Cl_CMB_model_from_manager(manager)
        cmb_map = mock.generate_map_cmb(
            Cl_cmb_model, config.nside, fixed_cmb=config.map_sim_pars.fixed_cmb
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

    if obsmats_loaded and dict_obsmats_func is not None:
        with Timer("filter-freq-maps"):
            for i_f, map_set_name in enumerate(dict_obsmats_func.keys()):
                logger.info(f"Filtering {config.frequencies[i_f]} channel")
                combined_freq_maps_beamed[i_f] = mock.apply_observation_matrix(
                    dict_obsmats_func[map_set_name], combined_freq_maps_beamed[i_f]
                )

    for i_f in range(len(config.frequencies)):
        combined_freq_maps[i_f] += noise_freq_maps[i_f]
        combined_freq_maps_beamed[i_f] += noise_freq_maps[i_f]

    combined_freq_maps = mask.apply_binary_mask(combined_freq_maps, binary_mask, unseen=False)
    combined_freq_maps_beamed = mask.apply_binary_mask(
        combined_freq_maps_beamed, binary_mask, unseen=False
    )

    timer.stop("simulate-one-sky-map")

    return noise_freq_maps, combined_freq_maps, combined_freq_maps_beamed


def load_obsmat(
    manager: DataManager,
    config: Config,
    obsmats_loaded: bool = False,
    dict_obsmats_func: dict | None = None,
):
    if config.map_sim_pars.filter_sims and not obsmats_loaded:
        logger.info("Loading observation matrices")
        with Timer(thread="load-obsmat"):
            dict_obsmats_func = mock.load_obseration_matrix(
                config.nside, config.map_sets, manager.get_osbmats_filenames()
            )
            return True, dict_obsmats_func
    elif obsmats_loaded:
        return True, dict_obsmats_func
    return False, None  # No obsmats needed


def save_sims(manager: DataManager, freq_maps_write):
    for i, fname in enumerate(manager.get_maps_filenames()):
        logger.debug(f"Saving simulated sky to {fname}")
        hp.write_map(
            fname,
            freq_maps_write[i],
            dtype=["float64", "float64", "float64"],
            overwrite=True,
        )


def save_noise_sims(manager: DataManager, noise_freq_maps_write, id_sim=0):
    # create the subdirectory for this realization
    manager.get_path_to_noise_maps_sub(id_sim).mkdir(parents=True, exist_ok=True)

    # save the maps
    for i, fname in enumerate(manager.get_noise_maps_filenames(sub=id_sim)):
        logger.debug(f"Saving noise simulation to {fname}")
        hp.write_map(
            fname,
            noise_freq_maps_write[i],
            dtype=["float64", "float64", "float64"],
            overwrite=True,
        )


def run_sim(args, id_sim=None, obsmats_loaded=False, dict_obsmats_func=None):
    if id_sim is None or (
        args.config and args.noise_only and args.Nsims
    ):  # Running only one simulation or multiple noise only simulaiton from 1 config
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
    if args.noise_only:
        noise_freq_maps, _, _ = make_sims(manager, config, components=["noise"])
        save_noise_sims(manager, noise_freq_maps, id_sim or 0)  # TODO check this
        return obsmats_loaded, dict_obsmats_func
    obsmats_loaded, dict_obsmats_func = load_obsmat(
        manager, config, obsmats_loaded, dict_obsmats_func
    )
    noise_freq_maps, _, combined_freq_maps_beamed = make_sims(
        manager, config, obsmats_loaded, dict_obsmats_func
    )
    save_sims(manager, combined_freq_maps_beamed)
    save_noise_sims(manager, noise_freq_maps, id_sim or 0)  # TODO check this
    return obsmats_loaded, dict_obsmats_func


def main():
    parser = argparse.ArgumentParser(description="simplistic simulator")
    parser.add_argument("--config", type=Path, help="config file")
    parser.add_argument(
        "--config_root", type=Path, help="config file root (will be appended  by {id_sim:04d})"
    )
    parser.add_argument("--Nsims", type=int, help="Number of simulations performed")
    parser.add_argument(
        "--noise-only", action="store_true", help="generate noise-only sims and save them to disk"
    )
    parser.add_argument(
        "--nomultiproc", action="store_true", help="don't use multprocessing parallelisation"
    )
    args = parser.parse_args()

    if args.config and not args.noise_only and not args.Nsims:  # Prioritize --config if provided
        run_sim(args)
        return

    if args.config_root or (
        args.config and args.noise_only and args.Nsims
    ):  # Multiple simulations mode
        if not args.Nsims:
            logger.warning("Nsims not specified, will only run one ")
            Nsims = 1
        else:
            Nsims = args.Nsims

        num_workers = 1 if args.nomultiproc else min(mp.cpu_count(), Nsims)
        obsmats_loaded, dict_obsmats_func = run_sim(
            args, 0
        )  # run the first iteration to load obsmat if required
        if obsmats_loaded:
            num_workers = (
                1  # if obsmats loaded, we don't parallelize #TODO find a way to parallelize ?
            )
        logger.info(f"Using {num_workers} worker processes")
        if num_workers > 1:
            mp.set_start_method("spawn", force=True)  # Ensure a clean multiprocessing start
            with mp.Pool(num_workers) as pool:
                pool.starmap(
                    run_sim,
                    [
                        (args, id_sim, obsmats_loaded, dict_obsmats_func)
                        for id_sim in range(1, Nsims)
                    ],
                )
        else:
            for id_sim in range(1, Nsims):
                run_sim(args, id_sim, obsmats_loaded, dict_obsmats_func)
    else:
        # Default case: no arguments provided, run single simulation with example config
        run_sim(args)


if __name__ == "__main__":
    main()
