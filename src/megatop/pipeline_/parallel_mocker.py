import argparse
from pathlib import Path

import healpy as hp
import numpy as np

from megatop import Config, DataManager
from megatop.utils import Timer, logger, mock


def make_sims(
    manager: DataManager,
    config: Config,
    components: str | list[str] = "all",
    dict_obsmats_func=None,
):
    timer = Timer()

    # create the directory for the maps
    manager.path_to_maps.mkdir(parents=True, exist_ok=True)

    timer.start("sim")
    binary_mask = hp.read_map(manager.path_to_binary_mask)
    fsky_binary = sum(binary_mask) / len(binary_mask)

    if components == "all":
        components = ["cmb", "fg", "noise"]

    if "noise" in components:
        # Creating noise maps
        timer.start("noise")

        if config.noise_sim_pars.noise_option == "white_noise":
            logger.info("Simulation has white noise only")
            # TODO: refactor to use config (requires changes in utils/mock.py)
            _, map_white_noise_levels = mock._get_noise(config, fsky_binary)
            noise_freq_maps = mock._get_noise_map_from_white_noise(manager, map_white_noise_levels)

        elif config.noise_sim_pars.noise_option == "no_noise":
            logger.info("Simulation has NO NOISE")
            noise_freq_maps = np.zeros((len(config.frequencies), 3, hp.nside2npix(config.nside)))

        elif config.noise_sim_pars.noise_option == "noise_spectra":
            logger.info("Simulation has noise from full spectra")
            n_ell, _ = mock._get_noise(config, fsky_binary)
            noise_freq_maps = mock._get_noise_map_from_noise_spectra(manager, n_ell)
        # elif meta.noise_sim_pars["noise_option"] == "MSS2":
        #     noise_maps = []
        #     print(
        #         "WARNING: When using MSS2 as noise_option, the nhits option and noise lvl will come from the noise_cov_pars. \n\
        #           noise_sims_pars.include_nhits should be put to FALSE. Otherwise the nhits will be applied twce and the noise std might be wrong."
        #     )
        #     for map_name in meta.maps_list:
        #         noise_maps.append(MakeNoiseMapsNhitsMSS2(meta, map_name, verbose=args.verbose).tolist())
        #     noise_maps = np.array(noise_maps, dtype=object)

        logger.debug(f"Noise maps has shape {noise_freq_maps.shape}")
        timer.stop("noise", "Computing noise maps")
    else:
        noise_freq_maps = np.zeros((len(config.frequencies), 3, hp.nside2npix(config.nside)))

    if "cmb" in components:
        # Performing the CMB simulation with synfast
        timer.start("cmb")
        logger.info("Computing CMB map from fiducial spectra")

        Cl_cmb_model = mock._get_Cl_CMB_model_from_manager(manager)
        cmb_map = mock._generate_map_cmb(config, Cl_cmb_model)

        logger.debug(f"CMB map has shape {cmb_map.shape}")
        timer.stop("cmb", "Computing CMB map")
    else:
        cmb_map = np.zeros((3, hp.nside2npix(config.nside)))

    if "fg" in components:
        # Generating pysm foreground simulations
        timer.start("fg")
        logger.info(f"Generating pysm sky {config.sky_model}")

        fg_freq_maps = mock._generate_map_fgs_pysm(config)

        logger.debug(f"Foreground map has shape {fg_freq_maps.shape}")
        timer.stop("fg", "Generating foreground map")
    else:
        fg_freq_maps = np.zeros((len(config.frequencies), 3, hp.nside2npix(config.nside)))

    combined_freq_maps = np.zeros((len(config.frequencies), 3, hp.nside2npix(config.nside)))
    combined_freq_maps_beamed = np.zeros((len(config.frequencies), 3, hp.nside2npix(config.nside)))

    if components == ["noise"]:
        noise_freq_maps[..., np.where(binary_mask == 0)[0]] = 0
        timer.stop("sim", "Simulating one sky (noise only)")
        return noise_freq_maps, None, None

    timer.start("beam")
    for i_f, _f in enumerate(config.frequencies):
        combined_freq_maps[i_f] = cmb_map + fg_freq_maps[i_f]
        combined_freq_maps_beamed[i_f] = mock._beam_winpix_correction(
            config, cmb_map + fg_freq_maps[i_f], config.beams[i_f]
        )
    timer.stop("beam", "Beaming frequency maps")

    if dict_obsmats_func is not None:
        timer.start("filtering")
        for i_f, map_set_name in enumerate(dict_obsmats_func.keys()):
            logger.info(f"Filtering {config.frequencies[i_f]} channel")
            combined_freq_maps_beamed[i_f] = mock.filter_obsmat(
                dict_obsmats_func[map_set_name], combined_freq_maps_beamed[i_f]
            )
        timer.stop("filtering", "Filtering frequency maps")

    timer.start("mask")

    for i_f, _f in enumerate(config.frequencies):
        combined_freq_maps[i_f] += noise_freq_maps[i_f]
        combined_freq_maps_beamed[i_f] += noise_freq_maps[i_f]
    # Applying binary mask to all products: #TODO move to utils ?
    combined_freq_maps[..., np.where(binary_mask == 0)[0]] = 0  # hp.UNSEEN
    combined_freq_maps_beamed[..., np.where(binary_mask == 0)[0]] = 0  # hp.UNSEEN
    timer.stop("mask", "Masking product with binary mask")
    timer.stop("sim", "Simulating one sky")

    return noise_freq_maps, combined_freq_maps, combined_freq_maps_beamed


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


def main():
    parser = argparse.ArgumentParser(description="simplistic simulator")
    parser.add_argument(
        "--config_root", type=Path, help="config file root (will be appended  by {id_sim:04d})"
    )
    parser.add_argument("--Nsims", type=int, help="Number of simulations performed")
    parser.add_argument(
        "--noise-only", action="store_true", help="generate noise-only sims and save them to disk"
    )
    parser.add_argument("--obsmat", action="store_true", help="filter the simulated maps")
    args = parser.parse_args()
    try:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.rank
        # root = 0

    except ImportError:
        logger.info("Could not find MPI. Proceeding without.")

        comm = None
        # root = 0
        rank = 0
        size = 1

    if args.obsmat:
        logger.info("Not using MPI with obsmat")
        comm = None
        # root = 0
        rank = 0
        size = 1

    # MemoryUsage(f"rank = {rank} ")

    logger.info(f"rank = {rank}, size = {size}")

    int_nreal = 1 if args.Nsims is None else args.Nsims
    realisation_list = np.arange(int_nreal)

    # splitting the list of simulation between the ranks of the process:
    rank_realisation_list = np.array_split(realisation_list, size)[rank]

    if args.obsmat:
        fname_config = args.config_root.with_name(args.config_root.name + "_0000.yml")
        config = Config.from_yaml(fname_config)
        manager = DataManager(config)
        manager.dump_config()
        timer = Timer()
        timer.start("loading obsmat")
        dict_obsmats_func = mock.load_obsmat(config, manager)
        timer.stop("loading obsmat", "Loading the obsernation matrices")
    else:
        dict_obsmats_func = None

    for id_realisation in rank_realisation_list:
        fname_config = args.config_root.with_name(
            args.config_root.name + f"_{id_realisation:04d}.yml"
        )
        logger.info(f"[{rank}] Reading config file {fname_config}")
        config = Config.from_yaml(fname_config)
        manager = DataManager(config)
        manager.dump_config()
        if args.noise_only:
            noise_freq_maps, _, _ = make_sims(manager, config, components=["noise"])
            save_noise_sims(manager, noise_freq_maps, id_realisation)
        else:
            noise_freq_maps, _, combined_freq_maps_beamed = make_sims(
                manager, config, components=["cmb", "noise"], dict_obsmats_func=dict_obsmats_func
            )
            save_sims(manager, combined_freq_maps_beamed)
            save_noise_sims(manager, noise_freq_maps, id_realisation)


if __name__ == "__main__":
    main()
