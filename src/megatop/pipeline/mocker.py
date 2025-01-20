import argparse
import os

import healpy as hp
import numpy as np

from megatop.utils import mock_utils
from megatop.utils.metadata_manager import BBmeta


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
            meta.logger.info("Simulation has white noise only")
            _, map_white_noise_levels = mock_utils.get_noise(meta, fsky_binary)
            noise_freqs_maps = mock_utils.get_noise_map_from_white_noise(
                meta, map_white_noise_levels
            )

        elif meta.noise_sim_pars["noise_option"] == "no_noise":
            meta.logger.info("Simulation has NO NOISE")
            noise_freqs_maps = np.zeros((len(meta.frequencies), 3, hp.nside2npix(meta.nside)))

        elif meta.noise_sim_pars["noise_option"] == "noise_spectra":
            meta.logger.info("Simulation has noise from full spectra")
            n_ell, _ = mock_utils.get_noise(meta, fsky_binary)
            noise_freqs_maps = mock_utils.get_noise_map_from_noise_spectra(meta, n_ell)
        # elif meta.noise_sim_pars["noise_option"] == "MSS2":
        #     noise_maps = []
        #     print(
        #         "WARNING: When using MSS2 as noise_option, the nhits option and noise lvl will come from the noise_cov_pars. \n\
        #           noise_sims_pars.include_nhits should be put to FALSE. Otherwise the nhits will be applied twce and the noise std might be wrong."
        #     )
        #     for map_name in meta.maps_list:
        #         noise_maps.append(MakeNoiseMapsNhitsMSS2(meta, map_name, verbose=args.verbose).tolist())
        #     noise_maps = np.array(noise_maps, dtype=object)

        meta.logger.debug(f"Noise maps has shape {noise_freqs_maps.shape}")
        meta.timer.stop("noise", meta.logger, "Computing noise maps")
    else:
        noise_freqs_maps = np.zeros((len(meta.frequencies), 3, hp.nside2npix(meta.nside)))

    if "cmb" in components:
        # Performing the CMB simulation with synfast
        meta.timer.start("cmb")
        meta.logger.info("Computing CMB map from fiducial spectra")

        Cl_cmb_model = mock_utils.get_Cl_CMB_model_from_meta(meta)
        cmb_map = mock_utils.generate_map_cmb(meta, Cl_cmb_model)

        meta.logger.debug(f"CMB map has shape {cmb_map.shape}")
        meta.timer.stop("cmb", meta.logger, "Computing CMB map")
    else:
        cmb_map = np.zeros((3, hp.nside2npix(meta.nside)))

    if "fg" in components:
        # Generating pysm foreground simulations
        meta.timer.start("fg")
        meta.logger.info(f"Generating pysm sky {(*meta.sky_model,)}")

        fg_freqs_maps = mock_utils.generate_map_fgs_pysm(meta)

        meta.logger.debug(f"Foreground map has shape {fg_freqs_maps.shape}")
        meta.timer.stop("fg", meta.logger, "Generating foreground map")
    else:
        fg_freqs_maps = np.zeros((len(meta.frequencies), 3, hp.nside2npix(meta.nside)))

    combined_freqs_maps = np.zeros((len(meta.frequencies), 3, hp.nside2npix(meta.nside)))
    combined_freqs_maps_beamed = np.zeros((len(meta.frequencies), 3, hp.nside2npix(meta.nside)))

    meta.timer.start("beam")
    for i_f, f in enumerate(meta.frequencies):
        combined_freqs_maps[i_f] = cmb_map + fg_freqs_maps[i_f] + noise_freqs_maps[i_f]
        combined_freqs_maps_beamed[i_f] = (
            mock_utils.beam_winpix_correction(
                meta, cmb_map + fg_freqs_maps[i_f], meta.beams_FWHM_arcmin[i_f]
            )
            + noise_freqs_maps[i_f]
        )
    meta.timer.stop("beam", meta.logger, "Beaming frequency maps")

    meta.timer.start("mask")

    # Applying binary mask to all products:
    combined_freqs_maps[..., np.where(binary_mask == 0)[0]] = 0  # hp.UNSEEN
    combined_freqs_maps_beamed[..., np.where(binary_mask == 0)[0]] = 0  # hp.UNSEEN
    meta.timer.stop("mask", meta.logger, "Masking product with binary mask")
    meta.timer.stop("sim", meta.logger, "Simulating one sky")

    return combined_freqs_maps, combined_freqs_maps_beamed


def save_sims(meta, freqs_maps_write):
    for i_f, map in enumerate(meta.maps_list):
        fname = os.path.join(meta.map_directory, meta.map_sets[map]["file_root"] + ".fits")
        hp.write_map(
            fname,
            freqs_maps_write[i_f],
            dtype=["float64", "float64", "float64"],
            overwrite=True,
        )


# def save_noise_maps(meta, noise_maps, id=0):
#     for i_f, map in enumerate(meta.maps_list):
#         fname = os.path.join(meta.mock_directory, meta.map_sets[map]["noise_root"] + ".fits")
#         hp.write_map(
#             fname, noise_maps[i_f], dtype=["float64", "float64", "float64"], overwrite=True
#         )
#     fname = os.path.join(meta.mock_directory, f"noise_sim_id_{id:04d}")
#     np.save(fname, noise_maps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="simplistic simulator")
    parser.add_argument("--globals", type=str, help="Path to yaml with global parameters")
    parser.add_argument(
        "--sim_id", type=int, help="Id of the simulation (useful fo noise covariance estimation)"
    )
    args = parser.parse_args()
    meta = BBmeta(args.globals)
    combined_freqs_maps, combined_freqs_maps_beamed = make_sims(meta)
    # if meta.noise_sim_pars["save_noise_sim"]:
    #     if args.sim_id is not None:
    #         save_noise_maps(meta, noise_maps, args.sim_id)
    #     else:
    #         save_noise_maps(meta, noise_maps)
    save_sims(meta, combined_freqs_maps_beamed)
