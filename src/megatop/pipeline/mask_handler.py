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
    #"http://pla.esac.esa.int/pla/aio/product-action?"
    #"MAP.MAP_ID=HFI_Mask_GalPlane-apo0_2048_R2.00.fits"
    "https://irsa.ipac.caltech.edu/data/Planck/release_2/"
    "ancillary-data/masks/HFI_Mask_GalPlane-apo0_2048_R2.00.fits"
)


# TODO: check the dtypes of products


def mask_handler(manager: DataManager, config: Config):
    experiments = set(map_set.exp_tag for map_set in config.map_sets)

    # Get the galactic mask (shared across all experiments)
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

    # Accumulateurs pour le masque joint (union binaire + hitmap combinée)
    binary_mask_union = np.zeros(hp.nside2npix(config.nside))
    common_norm_nhits_map_combined = np.zeros(hp.nside2npix(config.nside))

    # Loop over experiments: hitmap, binary mask, and analysis mask, each per experiment
    for exp in experiments:
        exp_map_sets = [m for m in config.map_sets if m.exp_tag == exp]

        with Timer(f"hitmap-{exp}"):
            if exp in config.masks_pars.uniform_coverage_exp_tags:
                logger.info(f"Using uniform (all-sky) coverage for {exp}")
                common_norm_nhits_map = np.ones(hp.nside2npix(config.nside), dtype=np.float32)
                for m in exp_map_sets:
                    hp.write_map(
                        manager.path_to_nhits_map(m),
                        common_norm_nhits_map,
                        dtype=np.float32,
                        overwrite=True,
                    )
            else:
                fwhm_arcmin_nhits = config.masks_pars.fwhm_arcmin_smooth_nhits
                if config.use_depth_maps:
                    logger.info(f"Loading depth maps for {exp}")
                    list_depthmapname = [m.depth_map_path for m in exp_map_sets]
                    depth_maps = mask.read_depth_maps(list_depthmapname, nside=config.nside)
                    norm_nhits_maps = mask.get_norm_smooth_nhits_from_depth(
                        depth_maps=depth_maps, fwhm_arcmin_nhits=fwhm_arcmin_nhits
                    )
                else:
                    logger.info(f"Loading nhits maps for {exp}")
                    list_hitmapname = [m.nhits_map_path for m in exp_map_sets]
                    nhits_maps = mask.read_nhits_maps(list_hitmapname, nside=config.nside)
                    norm_nhits_maps = mask.norm_smooth_nhits_maps(
                        nhits_maps=nhits_maps, fwhm_arcmin_nhits=fwhm_arcmin_nhits
                    )

                logger.info(f"Creating common nhits map for {exp}")
                common_norm_nhits_map = mask.get_common_nhits_map(
                    norm_nhits_maps, fwhm_arcmin_nhits=fwhm_arcmin_nhits
                )

                for i_m, m in enumerate(exp_map_sets):
                    hp.write_map(
                        manager.path_to_nhits_map(m),
                        norm_nhits_maps[i_m],
                        dtype=np.float32,
                        overwrite=True,
                    )

            hp.write_map(
                manager.path_to_common_nhits_map(exp),
                common_norm_nhits_map,
                dtype=np.float32,
                overwrite=True,
            )

        with Timer(f"binary-mask-{exp}"):
            threshold = config.masks_pars.binary_mask_zero_threshold
            logger.info(f"Thresholding binary map for {exp} with {threshold}")
            binary_mask = mask.get_binary_mask(common_norm_nhits_map, galactic_mask, threshold)
            hp.write_map(
                manager.path_to_binary_mask(exp), binary_mask, dtype=np.float32, overwrite=True
            )

        # 2e masque binaire avec un seuil à 1e-6
        with Timer(f"binary-mask-thr1e-6-{exp}"):
            binary_mask_alt = mask.get_binary_mask(common_norm_nhits_map, galactic_mask, 1e-6)
            hp.write_map(
                str(manager.path_to_binary_mask(exp)).replace(".fits", "_thr1e-6.fits"),
                binary_mask_alt,
                dtype=np.float32,
                overwrite=True,
            )

        # Accumulation pour le masque joint
        binary_mask_union = np.maximum(binary_mask_union, binary_mask)
        common_norm_nhits_map_combined = np.maximum(common_norm_nhits_map_combined, common_norm_nhits_map)

        #print(np.shape(common_norm_nhits_map))
        with Timer(f"apodize-custom-{exp}"):
            apod_radius = config.masks_pars.apod_radius
            apod_type = config.masks_pars.apod_type
            apodized_mask = mask.get_analysis_mask(
                common_norm_nhits_map, binary_mask, apod_radius_deg=apod_radius, apod_type=apod_type
            )
            hp.write_map(
                manager.path_to_analysis_mask(exp), apodized_mask, dtype=np.float32, overwrite=True
            )

        # Masque d'analyse joint : union des expériences, apodisé une seule fois
    with Timer("apodize-joint"):
        apod_radius = config.masks_pars.apod_radius
        apod_type = config.masks_pars.apod_type
        apodized_mask = mask.get_analysis_mask(
            common_norm_nhits_map_combined,
            binary_mask_union,
            apod_radius_deg=apod_radius,
            apod_type=apod_type,
        )
        hp.write_map(
            manager.path_to_joint_analysis_mask, apodized_mask, dtype=np.float32, overwrite=True
        )


def main():
    parser = argparse.ArgumentParser(description="Mask handler")
    parser.add_argument("--config", type=Path, required=True, help="config file")

    args = parser.parse_args()
    config = Config.load_yaml(args.config)
    manager = DataManager(config)

    world, rank, size = get_world()
    if rank == 0:
        manager.dump_config()
        manager.create_output_dirs(config.map_sim_pars.n_sim, config.noise_sim_pars.n_sim)

    mask_handler(manager, config)
    # test_mask(manager)


if __name__ == "__main__":
    main()