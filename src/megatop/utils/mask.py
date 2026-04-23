import os
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import healpy as hp
import numpy as np
import pymaster as nmt

from .logger import logger
from .timer import function_timer

SO_NOMINAL_HITMAP_URL = (
    "https://portal.nersc.gov/cfs/sobs/users/so_bb/norm_nHits_SA_35FOV_ns512.fits"
)

HEALPY_DATA_PATH = os.getenv("HEALPY_LOCAL_DATA", None)

# def get_norm_nhits_from_depth(depth_maps):
#     """
#     Compute normalised hitmap(s) from a depth map(s).
#     Parameters
#     ----------
#     depth_maps : array
#         input depth map (in muk_arcmin). shape is (..., npix)
#     """
#     if depth_maps.ndim > 1:
#         #norm = np.max(depth_maps**2, axis=1)
#         #norm_nhits = norm[:, np.newaxis, :] / (depth_maps**2)
#         #norm_nhits = np.max(depth_maps**2, axis=1)[..., np.newaxis] / depth_maps**2
#         #norm_nhits = np.where(depth_maps > 0, norm[:, np.newaxis, :] / (depth_maps**2), 0.0) # avoid division by 0?
#     else:
#         norm_nhits = np.max(depth_maps**2) / depth_maps**2
#     return hits_map_norm


def get_norm_smooth_nhits_from_depth(depth_maps, fwhm_arcmin_nhits):
    """
    Compute normalised and smoothed hitmap(s) from a depth map(s).
    Parameters
    ----------
    depth_maps : array
        input depth map (in muk_arcmin), shape is (..., npix)
    """
    nhits_map = np.zeros_like(depth_maps)
    valid = depth_maps > 0
    nhits_map[valid] = 1.0 / (depth_maps[valid] ** 2)
    return norm_smooth_nhits_maps(nhits_maps=nhits_map, fwhm_arcmin_nhits=fwhm_arcmin_nhits)


def smooth_mask(mask, fwhm_arcmin):
    """
    Smooth mask(s) with a (gaussian) beam.
    """
    if mask.ndim > 1:
        mask_smoothed = []
        for m in mask:
            mask_smoothed.append(
                hp.smoothing(m, fwhm=np.radians(fwhm_arcmin / 60.0), datapath=HEALPY_DATA_PATH)
            )
        mask_smoothed = np.array(mask_smoothed)
    else:
        mask_smoothed = hp.smoothing(
            mask, fwhm=np.radians(fwhm_arcmin / 60.0), datapath=HEALPY_DATA_PATH
        )
    mask_smoothed[mask_smoothed < 0] = 0
    return mask_smoothed


def read_depth_maps(list_depthmapname: list[Path], nside: int):
    """
    Read depth maps and ud_grade.
    """
    depth_maps = []
    for depthmapname in list_depthmapname:
        depth_maps.append(hp.ud_grade(hp.read_map(depthmapname, field=0), nside_out=nside))
    return np.array(depth_maps)


def read_nhits_maps(list_hitmapname: list[Path], nside: int):
    """
    Read hit maps and ud_grade.
    """
    nhits_maps = []
    SO_NOMINAL_NHITS = None
    for hitmapname in list_hitmapname:
        if hitmapname == "SO_nominal":
            if SO_NOMINAL_NHITS is None:
                try:
                    logger.info(f"Downloading nominal hit map from {SO_NOMINAL_HITMAP_URL}")
                    with urlopen(SO_NOMINAL_HITMAP_URL, timeout=10) as _:
                        # healpy can read directly from the URL
                        SO_NOMINAL_NHITS = hp.read_map(SO_NOMINAL_HITMAP_URL)
                        SO_NOMINAL_NHITS = hp.ud_grade(SO_NOMINAL_NHITS, nside, power=-2)
                except URLError:
                    logger.error("Nominal hitmap download failed")
                    logger.error("Exiting mask_handler without creating a mask")
                    sys.exit()
            nhits_maps.append(SO_NOMINAL_NHITS)
        else:
            nhits_maps.append(
                hp.ud_grade(hp.read_map(hitmapname, field=0), nside_out=nside, power=-2)
            )
    return np.array(nhits_maps)


def norm_smooth_nhits_maps(nhits_maps, fwhm_arcmin_nhits):
    """
    Normalize nhits_map(s)
    """
    if nhits_maps.ndim > 1:
        norm_nhits = nhits_maps / np.max(nhits_maps, axis=1)[..., np.newaxis]
    else:
        norm_nhits = nhits_maps / np.max(nhits_maps)
    return smooth_mask(mask=norm_nhits, fwhm_arcmin=fwhm_arcmin_nhits)


def get_common_nhits_map(hit_maps, fwhm_arcmin_nhits):
    """
    Get the 'common' nhits map by mutpliplying the (normalised) nhits maps for all frequencies together.
    """
    common_nhits_map = (
        np.prod(hit_maps, axis=0) ** (1.0 / hit_maps.shape[0]) if hit_maps.ndim > 1 else hit_maps
    )
    return smooth_mask(common_nhits_map / np.max(common_nhits_map), fwhm_arcmin=fwhm_arcmin_nhits)


def get_binary_mask(common_hitmap, gal_mask, zero_threshold):
    """
    Compute a binary mask by thresholding the common_hitmap and galactic mask.
    """
    # TODO: Include PS mask
    binary_mask = np.zeros_like(common_hitmap)
    binary_mask[common_hitmap > zero_threshold] = 1
    return binary_mask * gal_mask


def get_analysis_mask(common_hitmap, binary_mask, apod_radius_deg, apod_type):
    """
    Create analysis mask and apodize it.
    """
    return nmt.mask_apodization((binary_mask * common_hitmap), apod_radius_deg, apotype=apod_type)


def random_src_mask(mask, nsrcs, mask_radius_arcmin):
    """
    Generate a modified version of a mask by randomly masking circular regions around selected points.

    Parameters
    ----------
    mask : array
        input mask on which random point sources will be substracted
    nrscs : int
        number of random sources to be masked
    mask_radius_arcmin: float
        radius (in arcmin) of the mask around each source

    Returns
    -------
    ps_mask: array
    """
    ps_mask = mask.copy()
    rng = np.random.default_rng()
    src_ids = rng.choice(np.where(mask == 1)[0], nsrcs)
    for src_id in src_ids:
        vec = hp.pix2vec(hp.get_nside(mask), src_id)
        disc = hp.query_disc(hp.get_nside(mask), vec, np.deg2rad(mask_radius_arcmin / 60))
        ps_mask[disc] = 0
    return ps_mask


def get_binary_mask_from_nhits(nhits_map, nside, zero_threshold=1e-3):
    """
    Generate a binary mask from a nhits by setting to zero pixels below a certain threshold and to 1 others.

    Parameters
    ----------
    nhits_map : array
        maps of the nhits
    nside : int
        nside of the output mask
    zero_threshold: float
        threshold for setting pixels to 0 and 1.

    Returns
    -------
    binary_mask: array
    """
    nhits_smoothed = hp.smoothing(
        hp.ud_grade(nhits_map, nside, power=-2, dtype=np.float64),
        fwhm=np.pi / 180,
        datapath=HEALPY_DATA_PATH,
    )
    nhits_smoothed[nhits_smoothed < 0] = 0
    nhits_smoothed /= np.amax(nhits_smoothed)
    binary_mask = np.zeros_like(nhits_smoothed)
    binary_mask[nhits_smoothed > zero_threshold] = 1

    return binary_mask


def get_apodized_mask_from_nhits(
    nhits_map,
    nside,
    galactic_mask=None,
    point_source_mask=None,
    zero_threshold=1e-3,
    apod_radius=10.0,
    apod_radius_point_source=4.0,
    apod_type="C1",
    no_nhits_rescaling=False,
):
    """
        Produce an appropriately apodized mask from an nhits map as used in
        the BB pipeline paper (https://arxiv.org/abs/2302.04276).

        Procedure:
        * Make binary mask by smoothing, normalizing and thresholding nhits map
        * (optional) multiply binary mask by galactic mask
        * Apodize (binary * galactic)
        * (optional) multiply (binary * galactic) with point source mask
        * (optional) apodize (binary * galactic * point source)
        * Multiply everything by (smoothed) nhits map (if no_nhits_rescaling is False)

        Parameters
        ----------
        nhits_map : array
            maps of the nhits
        nside : int
            nside of the output mask.
        galactic_mask : array, optional
            galactic mask to apply.
        point_source_mask : array, optional
            point source mask to apply.
        zero_threshold : float, optional
            Threshold below which nhits values are set to zero.
        apod_radius : float, optional
            Apodization radius for the galactic mask (in degrees).
        apod_radius_point_source : float, optional
            Apodization radius for the point source mask (in degrees).
        apod_type : str, optional
            Type of apodization, default is "C1".
        no_nhits_rescaling : bool, optional
            If True, the apodized binary mask outputed is not resacled by nhits . Default is False.

    #     Returns
    #     -------
    #     array
    #         The apodized mask.
    #"""
    # Get binary mask
    binary_mask = get_binary_mask_from_nhits(nhits_map, nside, zero_threshold)

    # Multiply by Galactic mask
    if galactic_mask is not None:
        binary_mask *= hp.ud_grade(galactic_mask, nside)

    # Apodize the binary mask
    binary_mask = nmt.mask_apodization(binary_mask, apod_radius, apotype=apod_type)

    # Multiply with point source mask
    if point_source_mask is not None:
        binary_mask *= hp.ud_grade(point_source_mask, nside)
        binary_mask = nmt.mask_apodization(binary_mask, apod_radius_point_source, apotype=apod_type)
    if no_nhits_rescaling:
        return binary_mask
    return nhits_map * binary_mask


def get_spin_derivatives(map):
    """
    First and second spin derivatives of a given spin-0 map.
    Parameters
    ----------
    map : array
        Input spin-0 map in HEALPix format.

    Returns
    -------
    tuple of arrays
        First and second spin derivatives of the input map.
    """
    nside = hp.npix2nside(np.shape(map)[-1])
    ell = np.arange(3 * nside)
    alpha1i = np.sqrt(ell * (ell + 1.0))
    alpha2i = np.sqrt((ell - 1.0) * ell * (ell + 1.0) * (ell + 2.0))
    first = hp.alm2map(hp.almxfl(hp.map2alm(map, datapath=HEALPY_DATA_PATH), alpha1i), nside=nside)
    second = hp.alm2map(hp.almxfl(hp.map2alm(map, datapath=HEALPY_DATA_PATH), alpha2i), nside=nside)

    return first, second


def get_fsky(nhits_map, binary_mask, analysis_mask):
    fsky_nhits = np.mean(nhits_map)
    fsky_binary = np.mean(binary_mask)
    fsky_analysis = np.mean(analysis_mask)
    return fsky_nhits, fsky_binary, fsky_analysis


@function_timer("apply-binary-mask")
def apply_binary_mask(maps, binary_mask, unseen=False):
    # TODO the masking is done in place, needed or could be done differently ?
    if unseen:
        maps[..., np.where(binary_mask == 0)[0]] = hp.UNSEEN
    else:
        maps[..., np.where(binary_mask == 0)[0]] = 0.0
    return maps
