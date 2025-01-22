import argparse
import os
import urllib.request

import healpy as hp
import numpy as np

from megatop.utils import BBmeta
from megatop.utils.mask_utils import (
    get_apodized_mask_from_nhits,
    get_binary_mask_from_nhits,
    get_spin_derivatives,
    random_src_mask,
)


def mask_handler(meta):
    """ """
    mask_dir = meta.mask_directory
    os.makedirs(mask_dir, exist_ok=True)

    timeout_seconds = 300  # Set the timeout [sec] for the socket

    ### Get nhits map
    # If we don't use a custom nhits map, work with the nominal nhits map downloaded nominal hits map from URL
    meta.timer.start("nhits")
    urlpref = "https://portal.nersc.gov/cfs/sobs/users/so_bb/"
    url = f"{urlpref}norm_nHits_SA_35FOV_ns512.fits"
    meta.logger.info(f"Downloading nominal hit map from {url}")
    with (
        urllib.request.urlopen(url, timeout=timeout_seconds) as response,
        open("temp.fits", "w+b") as f,
    ):
        f.write(response.read())
    nhits_nominal = hp.ud_grade(hp.read_map("temp.fits"), meta.nside, power=-2)
    os.remove("temp.fits")

    if not meta.use_input_nhits:
        meta.logger.info("Using nominal hit map for analysis")
        nhits = nhits_nominal
    else:
        if not os.path.exists(meta.masks["input_nhits_path"]):
            meta.logger.info("Could not find input nhits map.")
            # if meta.filtering_type == "toast": #TODO not implemented yet
            #     print("Get nhits map from provided TOAST schedule.")
            #     meta.get_nhits_map_from_toast_schedule()
            # else:
            meta.logger.error(f"Cannot find nhits file {meta.masks['input_nhits_path']}")
            raise FileNotFoundError
        meta.logger.info(
            f"Using custom hit mask for analysis from {meta.masks['input_nhits_path']}"
        )
        nhits = meta.read_hitmap()
    meta.save_hitmap(nhits)

    meta.timer.stop("nhits", meta.logger, "Getting hits map")

    # Generate binary survey mask from the hits map
    meta.timer.start("binary")
    meta.logger.info(
        f"Thresholding hit map with threshold {meta.masks['binary_mask_zero_threshold']}"
    )
    binary_mask = get_binary_mask_from_nhits(
        nhits, meta.nside, zero_threshold=meta.masks["binary_mask_zero_threshold"]
    )
    meta.save_mask("binary", binary_mask, overwrite=True)
    meta.timer.stop("binary", meta.logger, "Computing binary mask")

    meta.timer.start("apodize")
    meta.logger.info(
        f"Apodizing nominal mask to {meta.masks['apod_radius']}arcmin with {meta.masks['apod_type']} scheme."
    )
    nominal_mask = get_apodized_mask_from_nhits(
        nhits_nominal,
        meta.nside,
        galactic_mask=None,
        point_source_mask=None,
        zero_threshold=meta.masks["binary_mask_zero_threshold"],
        apod_radius=meta.masks["apod_radius"],
        apod_type=meta.masks["apod_type"],
    )
    first_nom, second_nom = get_spin_derivatives(nominal_mask)
    meta.timer.stop("apodize", meta.logger, "Computing nominal apodized mask")

    if not meta.use_input_nhits:
        # Make nominal apodized mask from the nominal hits map
        meta.logger.info("Using nominal mask as final analysis one.")
        final_mask = nominal_mask
        first = first_nom
        second = second_nom
    else:
        # Assemble custom analysis mask from hits map and point source mask
        # stored at disk, and Planck Galactic masks downloaded in-place.

        meta.logger.info(
            f"Assembling custom analysis mask using: {(*meta.masks['include_in_mask'],)}"
        )
        # Download Galactic mask
        if "galactic" in meta.masks["include_in_mask"]:
            meta.timer.start("galactic")
            meta.logger.info(f"Using planck {meta.masks['galactic_mask_mode']} galactic mask")
            # mask_p15_file = f"{mask_dir}/mask_planck2015.fits"
            mask_p15_file = (
                "/Users/benjaminberingue/Downloads/HFI_Mask_GalPlane-apo2_2048_R2.00.fits"
            )
            if not os.path.exists(mask_p15_file):
                urlpref = "http://pla.esac.esa.int/pla/aio/"
                urlpref = f"{urlpref}product-action?MAP.MAP_ID="
                url = f"{urlpref}HFI_Mask_GalPlane-apo0_2048_R2.00.fits"
                meta.logger.info(f"Downloading from {url}")
                with urllib.request.urlopen(url, timeout=timeout_seconds):
                    urllib.request.urlretrieve(url, filename=mask_p15_file)
            else:
                meta.logger.info(f"Read from {mask_p15_file}")

            # Save different galactic masks
            gal_keys = [
                "GAL020",
                "GAL040",
                "GAL060",
                "GAL070",
                "GAL080",
                "GAL090",
                "GAL097",
                "GAL099",
            ]
            id_key = gal_keys.index(meta.masks["galactic_mask_mode"].upper())
            fname = os.path.join(
                f"{mask_dir}",
                f"{meta.masks['galactic_mask_root']}_{meta.masks['galactic_mask_mode']}.fits",
            )
            gal_mask_p15 = hp.read_map(mask_p15_file, field=id_key)
            if not os.path.exists(fname):
                # Rotate in equatorial coordinates
                r = hp.Rotator(coord=["G", "C"])
                gal_mask_p15 = r.rotate_map_pixel(gal_mask_p15)
                gal_mask_p15 = hp.ud_grade(gal_mask_p15, meta.nside)
                gal_mask_p15 = np.where(gal_mask_p15 > 0.5, 1, 0)
                hp.write_map(fname, gal_mask_p15, overwrite=True, dtype=np.int32)
            meta.timer.stop("galactic", meta.logger, "Galactic mask projection")

        # Get point source mask
        if "point_source" in meta.masks["include_in_mask"]:
            meta.timer.start("ps_mask")
            # Load from disk if file exists
            if meta.use_input_point_source:
                ps_fname = meta.masks["input_point_source_path"]
                meta.logger.info(f"Using point source mask from {meta.input_point_source_path}")
                ps_mask = binary_mask * hp.ud_grade(hp.read_map(ps_fname), meta.nside)
                meta.timer.stop("ps_mask", meta.logger, "Load point source mask from disk")
            # Otherwise, generate random point source mask
            else:
                meta.logger.info(
                    f"Generating random point source mask: {meta.masks['mock_nsrcs']} sources, {meta.masks['mock_srcs_hole_radius']} arcmin radius"
                )
                ps_mask = random_src_mask(
                    binary_mask, meta.masks["mock_nsrcs"], meta.masks["mock_srcs_hole_radius"]
                )
                meta.save_mask("point_source", ps_mask, overwrite=True)
                meta.timer.stop("ps_mask", meta.logger, "Generate mock point source mask")

        # Add the masks
        galactic_mask = None
        point_source_mask = None
        if "galactic" in meta.masks["include_in_mask"]:
            galactic_mask = meta.read_mask("galactic")
        if "point_source" in meta.masks["include_in_mask"]:
            point_source_mask = meta.read_mask("point_source")

        # Combine, apodize, and hits-weight the masks
        meta.timer.start("final_mask")
        if "point_source" in meta.masks["include_in_mask"]:
            meta.logger.info(
                f"Apodizing final analysis mask to {meta.masks['apod_radius']}arcmin, {meta.masks['apod_radius_point_source']}arcmin around point sources, with {meta.masks['apod_type']} scheme."
            )
        else:
            meta.logger.info(
                f"Apodizing final analysis mask to {meta.masks['apod_radius']}arcmin with {meta.masks['apod_type']} scheme."
            )
        final_mask = get_apodized_mask_from_nhits(
            nhits,
            meta.nside,
            galactic_mask=galactic_mask,
            point_source_mask=point_source_mask,
            zero_threshold=meta.masks["binary_mask_zero_threshold"],
            apod_radius=meta.masks["apod_radius"],
            apod_radius_point_source=meta.masks["apod_radius_point_source"],
            apod_type=meta.masks["apod_type"],
        )

        # Make sure first two spin derivatives are bounded below twice the
        # respective global maximum values of the nominal analysis mask.
        # If not, issue warning.
        first, second = get_spin_derivatives(final_mask)

        meta.logger.info(
            f"Using custom mask. Its spin derivatives have global min and max of: {np.amin(first)}, {np.amax(first)} (first), {np.amin(second)}, {np.amax(second)} (second)"
        )
        meta.logger.info(
            f"For comparison, the nominal mask has: {np.amin(first_nom)}, {np.amax(first_nom)} (first nominal), {np.amin(second_nom)}, {np.amax(second_nom)} (second nominal)"
        )

        first_is_bounded = np.amax(first) < 2 * np.amax(first_nom) and np.amin(first) > 2 * np.amin(
            first_nom
        )
        second_is_bounded = np.amax(second) < 2 * np.amax(second_nom) and np.amin(
            second
        ) > 2 * np.amin(second_nom)

        if not (first_is_bounded and second_is_bounded):
            meta.logger.info(
                "WARNING: Your analysis mask may not be smooth enough, "
                "so B-mode purification could induce biases."
            )
        meta.timer.stop("final_mask", meta.logger, "Assembling final analysis mask")

    # Save analysis mask
    meta.save_mask("analysis", final_mask, overwrite=True)


def main():
    parser = argparse.ArgumentParser(description="simplistic simulator")
    parser.add_argument("--globals", type=str, help="Path to yaml with global parameters")
    args = parser.parse_args()
    meta = BBmeta(args.globals)
    mask_handler(meta)


if __name__ == "__main__":
    main()
