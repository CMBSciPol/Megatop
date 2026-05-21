import healpy as hp
import numpy as np


def truncate_alm(alm, lmax_new):
    """
    Truncate the spherical harmonic coefficients (alm) to a new lmax.
    Parameters
    ----------
    alm : array_like
        Input spherical harmonic coefficients, shape (..., N), where N = (lmax_old + 1) * (lmax_old + 2) / 2.
    lmax_new : int
        New maximum multipole to truncate to.
    Returns
    -------
    alm_new : array_like
        Truncated spherical harmonic coefficients, shape (..., M), where M = (lmax_new + 1) * (lmax_new + 2) / 2.
    """
    lmax_old = hp.Alm.getlmax(alm.shape[-1])
    assert lmax_new <= lmax_old, "New lmax must be <= original lmax"
    new_shape = (*alm.shape[:-1], hp.Alm.getsize(lmax_new))
    alm_new = np.zeros(new_shape, dtype=alm.dtype)

    for ell in range(lmax_new + 1):
        for m in range(ell + 1):
            i_old = hp.Alm.getidx(lmax_old, ell, m)
            i_new = hp.Alm.getidx(lmax_new, ell, m)
            alm_new[..., i_new] = alm[..., i_old]

    return alm_new


def set_alm_tozero_below_lmin(alm, lmin):
    """
    Truncate the spherical harmonic coefficients (alm) to a new lmax.
    Parameters
    ----------
    alm : array_like
        Input spherical harmonic coefficients, shape (..., N), where N = (lmax_old + 1) * (lmax_old + 2) / 2.
    lmax_new : int
        New maximum multipole to truncate to.
    Returns
    -------
    alm_new : array_like
        Truncated spherical harmonic coefficients, shape (..., M), where M = (lmax_new + 1) * (lmax_new + 2) / 2.
    """
    lmax = hp.Alm.getlmax(alm.shape[-1])
    assert lmin <= lmax, "lmin must be < lmax"

    for ell in range(lmin):
        for m in range(ell + 1):
            index = hp.Alm.getidx(lmax, ell, m)
            alm[..., index] = 0 + 0j

    return alm


def set_alm_tozero_above_lmax(alm, lmax):
    """
    Truncate the spherical harmonic coefficients (alm) to a new lmax.
    Parameters
    ----------
    alm : array_like
        Input spherical harmonic coefficients, shape (..., N), where N = (lmax_old + 1) * (lmax_old + 2) / 2.
    lmax_new : int
        New maximum multipole to truncate to.
    Returns
    -------
    alm_new : array_like
        Truncated spherical harmonic coefficients, shape (..., M), where M = (lmax_new + 1) * (lmax_new + 2) / 2.
    """
    lmax_input = hp.Alm.getlmax(alm.shape[-1])
    assert lmax_input >= lmax, "lmax must be <= lmax_input"

    for ell in range(lmax + 1, lmax_input + 1):
        for m in range(ell + 1):
            index = hp.Alm.getidx(lmax_input, ell, m)
            alm[..., index] = 0 + 0j

    return alm


def get_smooth_scale_cut(cut_scale, smoothing_scale, lmax, lmin=0):
    """
    Get a smooth scale cut function that transitions from 0 to 1 around the specified cut scale.
    Only implemented shape is tanh.
    Parameters    ----------
    cut_scale : float
        The multipole scale around which the transition occurs.
    smoothing_scale : float
        The width of the transition region in multipole space.
    lmax : int
        The maximum multipole to consider.
    lmin : int, optional
        The minimum multipole to consider (default is 0).
    Returns    -------
    smooth_cut : array_like
        An array of shape (lmax + 1,) containing the smooth scale cut values for each multipole.
    """
    ell = np.arange(lmax + 1)
    smooth_cut = 0.5 * (1 + np.tanh((ell - cut_scale) / smoothing_scale))
    smooth_cut[:lmin] = 0.0
    return smooth_cut


def cut_map_scales(freq_maps_input, cut_array, nside):
    freq_maps_cut = np.zeros_like(freq_maps_input)
    for f in range(freq_maps_input.shape[0]):
        alm_comp = hp.map2alm(
            [
                freq_maps_input[f, 0],
                freq_maps_input[f, 1],
                freq_maps_input[f, 2],
            ],
            lmax=3 * nside,
        )
        for s in range(alm_comp.shape[0]):
            hp.almxfl(alm_comp[s], cut_array, inplace=True)
        freq_maps_cut[f] = hp.alm2map(
            alm_comp, nside=nside, lmax=3 * nside, pol=True
        )  # removing temperature
    return freq_maps_cut
