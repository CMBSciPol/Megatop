import argparse
import sys
from pathlib import Path
from typing import get_args
from urllib.error import URLError
from urllib.request import urlopen

import healpy as hp
import numpy as np

from megatop import DataManager
from megatop.config import Config, ValidPlanckGalKey
from megatop.utils import Timer, logger
from megatop.utils.mask import (
    get_apodized_mask_from_nhits,
    get_binary_mask_from_nhits,
    get_spin_derivatives,
    random_src_mask,
)
from megatop.utils.mpi import get_world

SO_NOMINAL_HITMAP_URL = (
    "https://portal.nersc.gov/cfs/sobs/users/so_bb/norm_nHits_SA_35FOV_ns512.fits"
)

PLANCK_MASK_GALPLANE_URL = (
    "http://pla.esac.esa.int/pla/aio/product-action?"
    "MAP.MAP_ID=HFI_Mask_GalPlane-apo0_2048_R2.00.fits"
)


# TODO: check the dtypes of products


def mask_handler(manager: DataManager, config: Config):
    timer = Timer()
    mask_dir = manager.path_to_masks
    mask_dir.mkdir(parents=True, exist_ok=True)

    # Get nhits map

    with Timer("hitmap"):
        # we always download the nominal hit map for reference
        # TODO: just write the values of the first and second derivatives somewhere
        if not config.use_input_nhits:
            logger.info("Using nominal hit map for analysis")
            try:
                logger.info(f"Downloading nominal hit map from {SO_NOMINAL_HITMAP_URL}")
                with urlopen(SO_NOMINAL_HITMAP_URL, timeout=10) as _:
                    # healpy can read directly from the URL
                    hitmap = hp.read_map(SO_NOMINAL_HITMAP_URL)
                    hitmap = hp.ud_grade(hitmap, config.nside, power=-2)
            except URLError:
                logger.error("No custom hitmap provided and nominal hitmap download failed")
                logger.error("Exiting mask_handler without creating a mask")
                sys.exit()
        else:
            logger.info("Using custom hit mask for analysis")
            hitmap = hp.read_map(config.masks_pars.input_nhits_map)
            hitmap = hp.ud_grade(hitmap, config.nside, power=-2)

        hp.write_map(manager.path_to_nhits_map, hitmap, dtype=np.float32, overwrite=True)

    # Generate binary survey mask from the hits map

    with Timer("binary-mask"):
        threshold = config.masks_pars.binary_mask_zero_threshold
        logger.info(f"Thresholding hit map with {threshold = }")
        binary_mask = get_binary_mask_from_nhits(hitmap, config.nside, zero_threshold=threshold)
        hp.write_map(manager.path_to_binary_mask, binary_mask, dtype=np.float32, overwrite=True)

    # Get the galactic mask

    galactic_mask = None

    if config.use_input_nhits and config.masks_pars.include_galactic:
        timer.start("galactic-mask")

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
        timer.stop("galactic-mask")

    # Get the point sources mask

    ps_mask = None

    if config.use_input_nhits and config.masks_pars.include_sources:
        timer.start("point-sources")
        if config.use_input_point_sources:
            # Load from disk
            mask_path: Path = config.masks_pars.input_sources_mask  # pyright: ignore[reportAssignmentType]
            logger.info(f"Using point source mask from {mask_path}")
            ps_mask = hp.read_map(mask_path)
            ps_mask = hp.ud_grade(ps_mask, config.nside)
            ps_mask *= binary_mask
        else:
            # Otherwise, generate random point source mask
            n_sources = config.masks_pars.mock_nsources
            hole_radius = config.masks_pars.mock_sources_hole_radius
            logger.info(f"Generating mock sources mask with {n_sources = }, {hole_radius =} arcmin")
            ps_mask = random_src_mask(binary_mask, n_sources, hole_radius)

        hp.write_map(manager.path_to_sources_mask, ps_mask, dtype=np.float32, overwrite=True)
        timer.stop("point-sources")

    # Apodize the updated mask
    # TODO: why is the hitmap still needed?

    apod_radius = config.masks_pars.apod_radius
    apod_type = config.masks_pars.apod_type

    # --------------------------------------
    # TODO: from here
    with Timer("apodize-custom"):
        # Make custom apodized mask from input hitmap, galactic mask and point sources mask
        apodized_mask = get_apodized_mask_from_nhits(
            hitmap,
            config.nside,
            galactic_mask=galactic_mask,
            point_source_mask=ps_mask,
            zero_threshold=threshold,
            apod_radius=apod_radius,
            apod_type=apod_type,
        )

        # Make sure first two spin derivatives are bounded below twice the
        # respective global maximum values of the nominal analysis mask.
        # If not, issue warning.
        first_custom, second_custom = get_spin_derivatives(apodized_mask)
        first_min_custom, first_max_custom = np.min(first_custom), np.max(first_custom)
        second_min_custom, second_max_custom = np.min(second_custom), np.max(second_custom)

        logger.info(
            "Using custom mask. Its spin derivatives have global min and max of:\n"
            f"  {first_min_custom}, {first_max_custom} (first),\n"
            f"  {second_min_custom}, {second_max_custom} (second)"
        )

    # Save final mask
    hp.write_map(manager.path_to_analysis_mask, apodized_mask, dtype=np.float32, overwrite=True)


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


if __name__ == "__main__":
    main()
