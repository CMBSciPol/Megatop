import argparse
import os

import healpy as hp
import numpy as np

from megatop.utils import BBmeta, mock
from megatop.utils.logger import logger 


def make_sims(meta, components="all"):
    """ """
    meta.timer.start("sim")
    binary_mask = meta.read_mask("binary")
    fsky_binary = sum(binary_mask) / len(binary_mask)

    if components == "all":
        components = ["cmb", "fg", "noise"]

    if "noise" in components:
        # Creating noise maps
        meta.timer.start("noise")

        if meta.noise_sim_pars["noise_option"] == "white_noise":
            logger.info("Simulation has white noise only")
            _, map_white_noise_levels = mock.get_noise(meta, fsky_binary)
            noise_freq_maps = mock.get_noise_map_from_white_noise(meta, map_white_noise_levels)

        elif meta.noise_sim_pars["noise_option"] == "no_noise":
            logger.info("Simulation has NO NOISE")
            noise_freq_maps = np.zeros((len(meta.frequencies), 3, hp.nside2npix(meta.nside)))

        elif meta.noise_sim_pars["noise_option"] == "noise_spectra":
            logger.info("Simulation has noise from full spectra")
            n_ell, _ = mock.get_noise(meta, fsky_binary)
            noise_freq_maps = mock.get_noise_map_from_noise_spectra(meta, n_ell)
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
        meta.timer.stop("noise", "Computing noise maps")
    else:
        noise_freq_maps = np.zeros((len(meta.frequencies), 3, hp.nside2npix(meta.nside)))

    if "cmb" in components:
        # Performing the CMB simulation with synfast
        meta.timer.start("cmb")
        logger.info("Computing CMB map from fiducial spectra")

        Cl_cmb_model = mock.get_Cl_CMB_model_from_meta(meta)
        cmb_map = mock.generate_map_cmb(meta, Cl_cmb_model)

        logger.debug(f"CMB map has shape {cmb_map.shape}")
        meta.timer.stop("cmb", "Computing CMB map")
    else:
        cmb_map = np.zeros((3, hp.nside2npix(meta.nside)))

    if "fg" in components:
        # Generating pysm foreground simulations
        meta.timer.start("fg")
        logger.info(f"Generating pysm sky {(*meta.sky_model,)}")

        fg_freq_maps = mock.generate_map_fgs_pysm(meta)

        logger.debug(f"Foreground map has shape {fg_freq_maps.shape}")
        meta.timer.stop("fg", "Generating foreground map")
    else:
        fg_freq_maps = np.zeros((len(meta.frequencies), 3, hp.nside2npix(meta.nside)))

    combined_freq_maps = np.zeros((len(meta.frequencies), 3, hp.nside2npix(meta.nside)))
    combined_freq_maps_beamed = np.zeros((len(meta.frequencies), 3, hp.nside2npix(meta.nside)))

    if components == ["noise"]:
        noise_freq_maps[..., np.where(binary_mask == 0)[0]] = 0
        meta.timer.stop("sim", "Simulating one sky (noise only)")
        return noise_freq_maps, None

    meta.timer.start("beam")
    for i_f, _f in enumerate(meta.frequencies):
        combined_freq_maps[i_f] = cmb_map + fg_freq_maps[i_f] + noise_freq_maps[i_f]
        combined_freq_maps_beamed[i_f] = (
            mock.beam_winpix_correction(
                meta, cmb_map + fg_freq_maps[i_f], meta.beams_FWHM_arcmin[i_f]
            )
            + noise_freq_maps[i_f]
        )
    meta.timer.stop("beam", "Beaming frequency maps")

    meta.timer.start("mask")

    # Applying binary mask to all products: #TODO move to utils ?
    combined_freq_maps[..., np.where(binary_mask == 0)[0]] = 0  # hp.UNSEEN
    combined_freq_maps_beamed[..., np.where(binary_mask == 0)[0]] = 0  # hp.UNSEEN
    meta.timer.stop("mask", "Masking product with binary mask")
    meta.timer.stop("sim", "Simulating one sky")

    return combined_freq_maps, combined_freq_maps_beamed


def save_sims(meta, freq_maps_write):
    for i_f, map in enumerate(meta.maps_list):
        fname = os.path.join(meta.map_directory, meta.map_sets[map]["file_root"] + ".fits")
        logger.debug(f"Saving simulated sky to {fname}")
        hp.write_map(
            fname,
            freq_maps_write[i_f],
            dtype=["float64", "float64", "float64"],
            overwrite=True,
        )


def save_noise_sims(meta, noise_freq_maps_write, id_sim=0):
    for i_f, map in enumerate(meta.maps_list):
        fname = meta.get_noise_map_filename(map, id_sim=id_sim)
        logger.debug(f"Saving noise simulation to {fname}")
        hp.write_map(
            fname,
            noise_freq_maps_write[i_f],
            dtype=["float64", "float64", "float64"],
            overwrite=True,
        )


def main():
    parser = argparse.ArgumentParser(description="simplistic simulator")
    parser.add_argument("--globals", type=str, help="Path to yaml with global parameters")
    parser.add_argument(
        "--sim_id", type=int, help="Id of the simulation (useful fo noise covariance estimation)"
    )
    parser.add_argument(
        "--noise_only",
        action="store_true",
        help="Flag to generate noise_only sims and save them to disk.",
    )
    args = parser.parse_args()
    meta = BBmeta(args.globals)
    if args.noise_only:
        combined_freq_maps, _ = make_sims(meta, components=["noise"])
        save_noise_sims(meta, combined_freq_maps, args.sim_id)
    else:
        combined_freq_maps, combined_freq_maps_beamed = make_sims(meta)
        save_sims(meta, combined_freq_maps_beamed)


if __name__ == "__main__":
    main()
