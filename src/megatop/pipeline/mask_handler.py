import argparse
from pathlib import Path
from typing import get_args
from urllib.error import URLError
from urllib.request import urlopen

import healpy as hp
import numpy as np

from megatop import DataManager
from megatop.config import Config, ValidPlanckGalKey
from megatop.utils import Timer, logger, mask
from megatop.utils.mpi import get_world

PLANCK_MASK_GALPLANE_URL = (
    "http://pla.esac.esa.int/pla/aio/product-action?"
    "MAP.MAP_ID=HFI_Mask_GalPlane-apo0_2048_R2.00.fits"
)


# TODO: check the dtypes of products


def mask_handler(manager: DataManager, config: Config):
    mask_dir = manager.path_to_masks
    mask_dir.mkdir(parents=True, exist_ok=True)

    # Get nhits map

    with Timer("hitmap"):
        fwhm_arcmin_nhits = config.masks_pars.fwhm_arcmin_smooth_nhits
        if config.use_depth_maps:
            logger.info("Loading depth maps")
            list_depthmapname = [m.depth_map_path for m in config.map_sets]
            depth_maps = mask.read_depth_maps(list_depthmapname, nside=config.nside)
            norm_nhits_maps = mask.get_norm_smooth_nhits_from_depth(
                depth_maps=depth_maps, fwhm_arcmin_nhits=fwhm_arcmin_nhits
            )
        else:
            logger.info("Loading nhits maps")
            list_hitmapname = [m.nhits_map_path for m in config.map_sets]
            nhits_maps = mask.read_nhits_maps(list_hitmapname, nside=config.nside)
            norm_nhits_maps = mask.norm_smooth_nhits_maps(
                nhits_maps=nhits_maps, fwhm_arcmin_nhits=fwhm_arcmin_nhits
            )

        logger.info("Creating common nhits map from geometrical mean of individual nhits maps")
        common_norm_nhits_map = mask.get_common_nhits_map(
            norm_nhits_maps, fwhm_arcmin_nhits=fwhm_arcmin_nhits
        )
        hp.write_map(
            manager.path_to_common_nhits_map,
            common_norm_nhits_map,
            dtype=np.float32,
            overwrite=True,
        )

        for i_m, m in enumerate(config.map_sets):
            hp.write_map(
                manager.path_to_nhits_map(m), norm_nhits_maps[i_m], dtype=np.float32, overwrite=True
            )

    # Get the galactic mask
    with Timer("galmask"):
        galactic_mask = np.ones(hp.nside2npix(config.nside))

        if config.masks_pars.include_galactic:
            # Download Planck galactic mask
            gal_key = config.masks_pars.gal_key
            index = get_args(ValidPlanckGalKey).index(gal_key)
            logger.info(f"Using Planck {gal_key!r} galactic mask ({index = })")
            try:
                logger.info(f"Downloading mask from {PLANCK_MASK_GALPLANE_URL}")
                with urlopen(PLANCK_MASK_GALPLANE_URL) as _:
                    # read only the requested field
                    galactic_mask = hp.read_map(PLANCK_MASK_GALPLANE_URL, field=index)
            except URLError as e:
                msg = "Failed to acess URL for Planck galactic mask"
                logger.error(msg)
                raise RuntimeError(msg) from e
            # Rotate from galactic to equatorial coordinates
            r = hp.Rotator(coord=["G", "C"])
            galactic_mask = r.rotate_map_pixel(galactic_mask)
            galactic_mask = hp.ud_grade(galactic_mask, config.nside)
            galactic_mask = np.where(galactic_mask > 0.5, 1, 0)

        hp.write_map(manager.path_to_galactic_mask, galactic_mask, dtype=np.float32, overwrite=True)

    # Generate binary survey mask from the hits map and galactic mask

    with Timer("binary-mask"):
        threshold = config.masks_pars.binary_mask_zero_threshold
        logger.info(f"Thresholding binary map with {threshold}")
        binary_mask = mask.get_binary_mask(common_norm_nhits_map, galactic_mask, threshold)
        hp.write_map(manager.path_to_binary_mask, binary_mask, dtype=np.float32, overwrite=True)

    with Timer("apodize-custom"):
        # Make custom apodized mask from input hitmap, galactic mask and point sources mask
        apod_radius = config.masks_pars.apod_radius
        apod_type = config.masks_pars.apod_type
        apodized_mask = mask.get_analysis_mask(
            common_norm_nhits_map, binary_mask, apod_radius_deg=apod_radius, apod_type=apod_type
        )
        hp.write_map(manager.path_to_analysis_mask, apodized_mask, dtype=np.float32, overwrite=True)


# Get the point sources mask

# ps_mask = None

# if config.use_input_nhits and config.masks_pars.include_sources:
#     timer.start("point-sources")
#     if config.use_input_point_sources:
#         # Load from disk
#         mask_path: Path = config.masks_pars.input_sources_mask
#         logger.info(f"Using point source mask from {mask_path}")
#         ps_mask = hp.read_map(mask_path)
#         ps_mask = hp.ud_grade(ps_mask, config.nside)
#         ps_mask *= binary_mask
#     else:
#         # Otherwise, generate random point source mask
#         n_sources = config.masks_pars.mock_nsources
#         hole_radius = config.masks_pars.mock_sources_hole_radius
#         logger.info(f"Generating mock sources mask with {n_sources = }, {hole_radius =} arcmin")
#         ps_mask = random_src_mask(binary_mask, n_sources, hole_radius)

#     hp.write_map(manager.path_to_sources_mask, ps_mask, dtype=np.float32, overwrite=True)
#     timer.stop("point-sources")

# if config.masks_pars.DEBUG_output_apod_binary_mask:
#     # This apodized mask is NOT multiplied by the hitmap
#     # This is intended to be used in the harmonic component separation
#     # It should not be used when purification is needed.
#     with Timer("apodize-custom binary"):
#         apodized_binary_mask = get_apodized_mask_from_nhits(
#             hitmap,
#             config.nside,
#             galactic_mask=galactic_mask,
#             point_source_mask=ps_mask,
#             zero_threshold=threshold,
#             apod_radius=apod_radius,
#             apod_type=apod_type,
#             no_nhits_rescaling=True,  # do not multiply by hitmap
#         )

#     hp.write_map(
#         manager.path_to_apod_binary_mask, apodized_binary_mask, dtype=np.float32, overwrite=True
#     )


def main():
    parser = argparse.ArgumentParser(description="Mask handler")
    parser.add_argument("--config", type=Path, required=True, help="config file")

    args = parser.parse_args()
    config = Config.load_yaml(args.config)
    manager = DataManager(config)

    world, rank, size = get_world()
    if rank == 0:
        manager.dump_config()

    mask_handler(manager, config)
    # test_mask(manager)


if __name__ == "__main__":
    main()
