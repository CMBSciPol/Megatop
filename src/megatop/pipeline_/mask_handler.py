import argparse
from pathlib import Path
from typing import get_args

import healpy as hp
import numpy as np

from megatop.config import Config, ValidPlanckGalKey
from megatop.utils import Timer, logger
from megatop.utils.mask import (
    get_apodized_mask_from_nhits,
    get_binary_mask_from_nhits,
    get_spin_derivatives,
    random_src_mask,
)

SO_NOMINAL_HITMAP_URL = (
    "https://portal.nersc.gov/cfs/sobs/users/so_bb/norm_nHits_SA_35FOV_ns512.fits"
)

PLANCK_MASK_GALPLANE_URL = (
    "http://pla.esac.esa.int/pla/aio/product-action?"
    "MAP.MAP_ID=HFI_Mask_GalPlane-apo0_2048_R2.00.fits"
)


# TODO: check the dtypes of products


def mask_handler(config: Config):
    timer = Timer()
    mask_dir = config.path_to_masks
    mask_dir.mkdir(parents=True, exist_ok=True)

    # Get nhits map

    timer.start("hitmap")

    # we always download the nominal hit map for reference
    # TODO: just write the values of the first and second derivatives somewhere

    # healpy can read directly from the URL
    logger.info(f"Downloading nominal hit map from {SO_NOMINAL_HITMAP_URL}")
    nominal_hitmap = hp.read_map(SO_NOMINAL_HITMAP_URL)
    nominal_hitmap = hp.ud_grade(nominal_hitmap, config.nside, power=-2)

    if config.use_input_nhits:
        # TODO: write test that confirms that the input path can not be None in this branch
        logger.info("Using custom hit mask for analysis")
        hitmap = hp.read_map(config.masks_pars.input_nhits_map)
        hitmap = hp.ud_grade(hitmap, config.nside, power=-2)
    else:
        logger.info("Using nominal hit map for analysis")
        hitmap = nominal_hitmap

    hp.write_map(config.path_to_nhits_map, hitmap, dtype=np.float32, overwrite=True)
    timer.stop("hitmap", "Getting hits map")

    # Generate binary survey mask from the hits map

    timer.start("binary")
    threshold = config.masks_pars.binary_mask_zero_threshold
    logger.info(f"Thresholding hit map with {threshold = }")
    binary_mask = get_binary_mask_from_nhits(hitmap, config.nside, zero_threshold=threshold)
    hp.write_map(config.path_to_binary_mask, binary_mask, dtype=np.float32, overwrite=True)
    timer.stop("binary", "Binary mask")

    # Get the galactic mask

    galactic_mask = None

    if config.use_input_nhits and config.masks_pars.include_galactic:
        timer.start("galactic")

        # Download Planck galactic mask
        gal_key = config.masks_pars.gal_key
        index = get_args(ValidPlanckGalKey).index(gal_key)
        logger.info(f"Using Planck {gal_key!r} galactic mask ({index = })")
        logger.info(f"Downloading mask from {PLANCK_MASK_GALPLANE_URL}")
        galactic_mask = hp.read_map(PLANCK_MASK_GALPLANE_URL, field=index)

        # Rotate from galactic to equatorial coordinates
        r = hp.Rotator(coord=["G", "C"])
        galactic_mask = r.rotate_map_pixel(galactic_mask)
        galactic_mask = hp.ud_grade(galactic_mask, config.nside)
        galactic_mask = np.where(galactic_mask > 0.5, 1, 0)

        hp.write_map(config.path_to_galactic_mask, galactic_mask, dtype=np.float32, overwrite=True)
        timer.stop("galactic", "Galactic mask")

    # Get the point sources mask

    ps_mask = None

    if config.use_input_nhits and config.masks_pars.include_sources:
        timer.start("point_sources")
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

        hp.write_map(config.path_to_sources_mask, ps_mask, dtype=np.float32, overwrite=True)
        timer.stop("point_sources", "Point sources mask")

    # Apodize the updated mask
    # TODO: why is the hitmap still needed?

    apod_radius = config.masks_pars.apod_radius
    apod_type = config.masks_pars.apod_type

    timer.start("apodize")
    logger.info(f"Apodizing nominal mask to {apod_radius = } arcmin with {apod_type = }")

    nominal_mask = get_apodized_mask_from_nhits(
        nominal_hitmap,
        config.nside,
        galactic_mask=None,
        point_source_mask=None,
        zero_threshold=threshold,
        apod_radius=apod_radius,
        apod_type=apod_type,
    )
    first_nom, second_nom = get_spin_derivatives(nominal_mask)
    first_min_nom, first_max_nom = np.min(first_nom), np.max(first_nom)
    second_min_nom, second_max_nom = np.min(second_nom), np.max(second_nom)

    timer.stop("apodize", "Apodized nominal mask")

    # --------------------------------------
    # TODO: from here

    if not config.use_input_nhits:
        # Make nominal apodized mask from the nominal hits map
        logger.info("Using nominal mask as final analysis one.")
        apodized_mask = nominal_mask
    else:
        timer.start("apodize_custom")

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
        logger.info(
            "For comparison, the nominal mask has:\n"
            f"  {first_min_nom}, {first_max_nom} (first nominal),\n"
            f"  {second_min_nom}, {second_max_nom} (second nominal)"
        )

        first_is_bounded = (
            2 * first_min_nom < first_min_custom and first_max_custom < 2 * first_max_nom
        )
        second_is_bounded = (
            2 * second_min_nom < second_min_custom and second_max_custom < 2 * second_max_nom
        )

        if not (first_is_bounded and second_is_bounded):
            logmsg = (
                "WARNING: Your analysis mask may not be smooth enough, "
                "so B-mode purification could induce biases."
            )
            logger.warning(logmsg)

        timer.stop("apodize_custom", "Apodize custom mask")

    # Save final mask
    hp.write_map(config.path_to_analysis_mask, apodized_mask, dtype=np.float32, overwrite=True)


def main():
    parser = argparse.ArgumentParser(description="Mask handler")
    parser.add_argument("--config", type=Path, help="config file")
    args = parser.parse_args()
    config = Config.from_yaml(args.config)
    config.dump()
    mask_handler(config)


if __name__ == "__main__":
    main()
