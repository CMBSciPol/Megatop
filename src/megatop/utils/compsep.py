import healpy as hp


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
