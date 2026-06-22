import io
import os
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import healpy as hp
import numpy as np
import pymaster as nmt
from pixell import enmap
from pixell import utils as pu

import megatop.utils.harmonic as hu
from megatop.config import SO_NOMINAL

from .logger import logger
from .timer import function_timer

SO_NOMINAL_HITMAP_URL = (
    "https://portal.nersc.gov/cfs/sobs/users/so_bb/norm_nHits_SA_35FOV_ns512.fits"
)

# Override the SO nominal hitmap source with a local file (e.g. a small fixture
# for CI/tests). When set, no network access is performed.
SO_NOMINAL_HITMAP_PATH = os.getenv("SO_NOMINAL_HITMAP_PATH", None)

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
    mask_smoothed = hu.smooth(mask, fwhm_arcmin)
    mask_smoothed[mask_smoothed < 0] = 0
    return mask_smoothed


def read_depth_maps(list_depthmapname: list[Path], nside: int):
    """
    Read depth maps and ud_grade. Maps are assumed equatorial (celestial).
    """
    depth_maps = []
    for depthmapname in list_depthmapname:
        depth_maps.append(hp.ud_grade(hp.read_map(depthmapname, field=0), nside_out=nside))
    return np.array(depth_maps, dtype=np.float64)


def read_nhits_maps(list_hitmapname: list[Path], nside: int):
    """
    Read hit maps and ud_grade. Maps are assumed equatorial (celestial).
    """
    nhits_maps = []
    so_nominal_nhits = None
    for hitmapname in list_hitmapname:
        if hitmapname == SO_NOMINAL:
            if so_nominal_nhits is None:
                if SO_NOMINAL_HITMAP_PATH is not None:
                    logger.info(f"Reading nominal hit map from {SO_NOMINAL_HITMAP_PATH}")
                    so_nominal_nhits = hp.read_map(SO_NOMINAL_HITMAP_PATH)
                else:
                    try:
                        logger.info(f"Downloading nominal hit map from {SO_NOMINAL_HITMAP_URL}")
                        # Fetch once into memory (bounded by timeout), then let healpy
                        # parse the bytes — avoids a second, unbounded URL read.
                        with urlopen(SO_NOMINAL_HITMAP_URL, timeout=30) as resp:
                            so_nominal_nhits = hp.read_map(io.BytesIO(resp.read()))
                    except URLError:
                        logger.error("Nominal hitmap download failed")
                        logger.error("Exiting mask_handler without creating a mask")
                        sys.exit()
                so_nominal_nhits = hp.ud_grade(so_nominal_nhits, nside, power=-2)
            nhits_maps.append(so_nominal_nhits)
        else:
            nhits_maps.append(
                hp.ud_grade(hp.read_map(hitmapname, field=0), nside_out=nside, power=-2)
            )
    return np.array(nhits_maps, dtype=np.float64)


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
    Create analysis mask and apodize it (HEALPix, via NaMaster C1/C2/Smooth).
    """
    return nmt.mask_apodization((binary_mask * common_hitmap), apod_radius_deg, apotype=apod_type)


def _apod_profile_c1(x):
    r"""NaMaster ``C1`` apodization profile, $f(x) = x - \sin(2\pi x)/(2\pi)$."""
    x = np.clip(x, 0.0, 1.0)
    return x - np.sin(2.0 * np.pi * x) / (2.0 * np.pi)


# CAR apodization profiles, keyed by NaMaster `apotype`. `enmap.apod_mask`
# applies the profile to the geodesic edge distance r/width, which equals
# NaMaster's chordal x to small-angle order.
# `Smooth` is a Gaussian convolution, not a profile, so it has no CAR analogue.
_CAR_APOD_PROFILES = {"C1": _apod_profile_c1, "C2": enmap.apod_profile_cos}


def get_analysis_mask_car(common_hitmap, binary_mask, apod_radius_deg, apod_type="C2"):
    """Create and apodize a CAR analysis mask.

    Uses ``pixell.enmap.apod_mask`` (a distance-transform taper of
    ``apod_radius_deg`` degrees) instead of NaMaster's apodization, which is
    HEALPix-only. ``apod_type`` selects the NaMaster-equivalent profile (``C1``
    or ``C2``); ``Smooth`` is unsupported on CAR. The apodized binary mask is
    then weighted by the hit map to match the HEALPix ``apodize(binary * nhits)``
    intent.
    """
    try:
        profile = _CAR_APOD_PROFILES[apod_type]
    except KeyError:
        msg = f"Unsupported CAR apod_type {apod_type!r}; choose one of {sorted(_CAR_APOD_PROFILES)}"
        raise ValueError(msg) from None
    apodized = enmap.apod_mask(binary_mask, width=apod_radius_deg * pu.degree, profile=profile)
    return apodized * common_hitmap


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
    nhits_smoothed = hu.smooth(hp.ud_grade(nhits_map, nside, power=-2, dtype=np.float64), 60.0)
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


def get_spin_derivatives(map, lmax=None):
    """
    First and second spin derivatives of a given spin-0 map.
    Parameters
    ----------
    map : array
        Input spin-0 map (HEALPix ``(npix,)`` ndarray or CAR ``(ny, nx)``
        enmap). The landscape is inferred from the input type.
    lmax : int, optional
        Band limit for the SHT round-trip. Required for CAR (no ``nside`` to
        derive one); defaults to ``3 * nside - 1`` for HEALPix.

    Returns
    -------
    tuple of arrays
        First and second spin derivatives of the input map.
    """
    if isinstance(map, enmap.ndmap):
        if lmax is None:
            raise ValueError("lmax is required for CAR maps")
        target = {"shape": map.shape, "wcs": map.wcs}
    else:
        nside = hp.npix2nside(np.shape(map)[-1])
        lmax = 3 * nside - 1 if lmax is None else lmax
        target = {"nside": nside}
    ell = np.arange(lmax + 1)
    alpha1i = np.sqrt(ell * (ell + 1.0))
    alpha2i = np.sqrt((ell - 1.0) * ell * (ell + 1.0) * (ell + 2.0))
    alm = hu.map2alm(map, spin=0, lmax=lmax)
    first = hu.alm2map(hu.almxfl(alm, alpha1i), spin=0, **target)
    second = hu.alm2map(hu.almxfl(alm, alpha2i), spin=0, **target)

    return first, second


def wmoment(field, p):
    r"""Solid-angle-weighted moment of a weight/mask map: $\int W^p\,d\Omega / 4\pi$.

    HEALPix pixels are equal-area, so this reduces to `mean(field**p)`. CAR pixels
    vary in area as $\cos(\mathrm{dec})$, so the moment must be area-weighted via
    `enmap.pixsizemap`; a plain `mean` is wrong there.

    Common uses (with $W$ the normalized weight, `mask` binary):

    - `wmoment(W, 2)` — pseudo-$C_\ell$ amplitude / debias normalization
    - `wmoment(mask, 1)` — geometric (binary) sky fraction
    - `wmoment(W, 2)**2 / wmoment(W, 4)` — variance / effective-DOF fsky
      (Hivon et al. 2002, $w_2^2/w_4$)
    """
    if isinstance(field, enmap.ndmap):
        area = enmap.pixsizemap(field.shape, field.wcs)
        return float(np.sum(area * field**p) / (4 * np.pi))
    return float(np.mean(field**p))


def fsky_effective(nhits):
    r"""Effective sky fraction $\langle \mathrm{nhits}\rangle$.

    Equivalent uniform-depth survey area: the V3p1 noise-amplitude input (the
    area the integration time is spread over). Smaller than geometric, because
    shallow edges discount the area.
    """
    return wmoment(nhits, 1)


def fsky_geom(binary_mask):
    r"""Geometric sky fraction $\langle \mathrm{mask}\rangle$.

    Solid-angle survey fraction; the debias factor for a pseudo-$C_\ell$ measured
    by `anafast` on a binary-masked map.
    """
    return wmoment(binary_mask, 1)


def fsky_w2(field):
    r"""Amplitude / debias normalization $\langle W^2\rangle$ (MCM row-sum)."""
    return wmoment(field, 2)


def fsky_dof(field):
    r"""Effective-DOF sky fraction for variance / error bars.

    Hivon et al. 2002 mode-count factor $f_{\rm sky}\,w_2^2/w_4 = \langle W^2\rangle^2 / \langle W^4\rangle$.
    Reduces to the geometric fraction for a binary mask; smaller for an apodized
    one (apodization loses modes).
    """
    return wmoment(field, 2) ** 2 / wmoment(field, 4)


@function_timer("apply-binary-mask")
def apply_binary_mask(maps, binary_mask, *, unseen=False):
    """Zero (or set ``hp.UNSEEN`` in) the pixels where ``binary_mask`` is 0.

    Pixel-agnostic: works for 1-D HEALPix masks ``(npix,)`` and 2-D CAR masks
    ``(ny, nx)`` via trailing boolean indexing over the map's pixel axes.
    Modifies ``maps`` in place and returns it.
    """
    bad = np.asarray(binary_mask) == 0
    maps[..., bad] = hp.UNSEEN if unseen else 0.0
    return maps
