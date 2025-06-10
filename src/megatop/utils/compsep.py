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
    new_shape = alm.shape[:-1] + (hp.Alm.getsize(lmax_new),)
    alm_new = np.zeros(new_shape, dtype=alm.dtype)

    for ell in range(lmax_new + 1):
        for m in range(ell + 1):
            i_old = hp.Alm.getidx(lmax_old, ell, m)
            i_new = hp.Alm.getidx(lmax_new, ell, m)
            alm_new[..., i_new] = alm[..., i_old]

    return alm_new
