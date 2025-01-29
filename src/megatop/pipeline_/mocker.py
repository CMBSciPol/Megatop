import argparse
from pathlib import Path

import healpy as hp
import numpy as np

from megatop import Config, DataManager
from megatop.utils import Timer, logger, mock


def make_sims(manager: DataManager, config: Config, components: str | list[str] = "all"):
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
        return noise_freq_maps, None

    timer.start("beam")
    for i_f, _f in enumerate(config.frequencies):
        combined_freq_maps[i_f] = cmb_map + fg_freq_maps[i_f] + noise_freq_maps[i_f]
        combined_freq_maps_beamed[i_f] = (
            mock._beam_winpix_correction(config, cmb_map + fg_freq_maps[i_f], config.beams[i_f])
            + noise_freq_maps[i_f]
        )
    timer.stop("beam", "Beaming frequency maps")

    timer.start("mask")

    # Applying binary mask to all products: #TODO move to utils ?
    combined_freq_maps[..., np.where(binary_mask == 0)[0]] = 0  # hp.UNSEEN
    combined_freq_maps_beamed[..., np.where(binary_mask == 0)[0]] = 0  # hp.UNSEEN
    timer.stop("mask", "Masking product with binary mask")
    timer.stop("sim", "Simulating one sky")

    return combined_freq_maps, combined_freq_maps_beamed


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
    parser.add_argument("--config", type=Path, help="config file")
    parser.add_argument(
        "--sim-id", type=int, help="Id of the simulation (useful fo noise covariance estimation)"
    )
    parser.add_argument(
        "--noise-only", action="store_true", help="generate noise-only sims and save them to disk"
    )
    args = parser.parse_args()
    if args.config is None:
        logger.warning("No config file provided, using example config")
        config = Config.get_example()
    else:
        config = Config.from_yaml(args.config)
    manager = DataManager(config)
    manager.dump_config()
    if args.noise_only:
        combined_freq_maps, _ = make_sims(manager, config, components=["noise"])
        save_noise_sims(manager, combined_freq_maps, args.sim_id)
    else:
        combined_freq_maps, combined_freq_maps_beamed = make_sims(manager, config)
        save_sims(manager, combined_freq_maps_beamed)


if __name__ == "__main__":
    main()
