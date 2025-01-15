import healpy as hp
import numpy as np 
import pymaster as nmt


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
    src_ids = np.random.choice(np.where(mask == 1)[0], nsrcs)
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
        hp.ud_grade(nhits_map, nside, power=-2, dtype=np.float64), fwhm=np.pi / 180
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
    * Multiply everything by (smoothed) nhits map

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

    Returns
    -------
    array
        The apodized mask.
    """
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
    first = hp.alm2map(hp.almxfl(hp.map2alm(map), alpha1i), nside=nside)
    second = hp.alm2map(hp.almxfl(hp.map2alm(map), alpha2i), nside=nside)

    return first, second