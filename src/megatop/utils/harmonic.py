# Adapted from SMARTIES (Simons Observatory Collaboration), MIT-licensed.
# Original source: harmonics.py at the SMARTIES project root.
"""Pixelization-agnostic spherical harmonic transforms.

Dispatches between HEALPIX (via ducc0) and CAR (via pixell.curvedsky) based
on the input map type. Public API takes keyword-only arguments and uses
native shapes for each pixelization: ``(..., npix)`` for HEALPIX and
``(..., ny, nx)`` for CAR.
"""

from __future__ import annotations

import ducc0
import healpy as hp
import numpy as np
from pixell import curvedsky, enmap

__all__ = [
    "alm2map",
    "almxfl",
    "anafast",
    "getlmax",
    "map2alm",
    "synfast",
]


def _is_car(x) -> bool:
    return isinstance(x, enmap.ndmap)


def getlmax(alm) -> int:
    """Infer ``lmax`` from a triangular-layout ``alm`` array.

    Args:
        alm: Spherical harmonic coefficients with the last axis storing the
            triangular ``(l, m)`` layout.

    Returns:
        Maximum multipole ``lmax`` consistent with ``alm.shape[-1]``.
    """
    return hp.Alm.getlmax(alm.shape[-1])


def _ducc_sht_kwargs(*, spin, nside, lmax, mmax=None):
    if mmax is None:
        mmax = lmax
    base = ducc0.healpix.Healpix_Base(nside, "RING")
    return {"spin": spin, "lmax": lmax, "mmax": mmax, **base.sht_info()}


def _alm2map_healpix(alms, *, spin, nside, lmax=None, mmax=None):
    """Thin ducc0 synthesis wrapper.

    ``alms`` must follow ducc0's ``([ntrans,] nmaps, nalm)`` convention:
    ``nmaps == 1`` for spin 0, ``nmaps == 2`` for spin > 0.
    """
    alm_lmax = getlmax(alms)
    if lmax is None:
        lmax = alm_lmax
    elif lmax > alm_lmax:
        raise ValueError(f"lmax={lmax} exceeds alm bandlimit {alm_lmax}")
    return ducc0.sht.synthesis(
        alm=alms,
        nthreads=0,
        **_ducc_sht_kwargs(spin=spin, nside=nside, lmax=lmax, mmax=mmax),
    )


def _map2alm_healpix(maps, *, spin, lmax=None, mmax=None):
    """Thin ducc0 adjoint-synthesis wrapper.

    ``maps`` must follow ducc0's ``([ntrans,] nmaps, npix)`` convention.
    """
    nside = hp.npix2nside(maps.shape[-1])
    if lmax is None:
        lmax = 3 * nside - 1
    weight = 4 * np.pi / (12 * nside**2)
    return ducc0.sht.adjoint_synthesis(
        map=maps * weight,
        nthreads=0,
        **_ducc_sht_kwargs(spin=spin, nside=nside, lmax=lmax, mmax=mmax),
    )


def _map2alm_healpix_iter(maps, *, spin, lmax=None, mmax=None, niter=0):
    nside = hp.npix2nside(maps.shape[-1])
    alm = _map2alm_healpix(maps, spin=spin, lmax=lmax, mmax=mmax)
    if niter == 0:
        return alm
    for _ in range(niter):
        residual = _alm2map_healpix(alm, spin=spin, nside=nside, lmax=lmax, mmax=mmax) - maps
        alm -= _map2alm_healpix(residual, spin=spin, lmax=lmax, mmax=mmax)
    return alm


def map2alm(maps, *, spin=0, lmax=None, mmax=None, niter=0):
    """Forward SHT, dispatching on pixelization.

    Args:
        maps: Input map. CAR shape ``(..., ny, nx)``; HEALPIX shape
            ``(..., npix)``. CAR inputs must be ``pixell.enmap.ndmap``.
        spin: Spin of the field (0 for T, 2 for (Q, U)).
        lmax: Bandlimit. Defaults to ``3 * nside - 1`` for HEALPIX and to
            the library default for CAR.
        mmax: Azimuthal bandlimit. Defaults to ``lmax``.
        niter: Jacobi iterations for HEALPIX adjoint synthesis (ignored for
            CAR).

    Returns:
        Spherical harmonic coefficients with the last axis storing the
        triangular ``(l, m)`` layout.
    """
    if _is_car(maps):
        return curvedsky.map2alm(maps, spin=spin, lmax=lmax, niter=niter)
    # ducc0 wants an explicit nmaps axis (1 for spin 0, 2 for spin > 0).
    work = maps[..., None, :] if spin == 0 else maps
    alm = _map2alm_healpix_iter(work, spin=spin, lmax=lmax, mmax=mmax, niter=niter)
    return alm[..., 0, :] if spin == 0 else alm


def alm2map(
    alms,
    *,
    spin=0,
    nside=None,
    shape=None,
    wcs=None,
    out=None,
    lmax=None,
    mmax=None,
):
    """Inverse SHT.

    Output pixelization is selected by which kwargs are given:

    - ``out`` (enmap) → CAR, written into ``out``.
    - ``shape`` and ``wcs`` → CAR, fresh enmap.
    - ``nside`` → HEALPIX (ducc0).

    Args:
        alms: Spherical harmonic coefficients.
        spin: Spin of the field (0 for T, 2 for (Q, U)).
        nside: HEALPIX resolution. Mutually exclusive with the CAR options.
        shape: CAR pixel shape. Used together with ``wcs``.
        wcs: CAR world coordinate system. Used together with ``shape``.
        out: Pre-allocated CAR enmap to write into. Mutually exclusive with
            ``nside``.
        lmax: Bandlimit. Defaults to inferred value from ``alms``.
        mmax: Azimuthal bandlimit. Defaults to ``lmax``.

    Returns:
        Pixel map. ``np.ndarray`` of shape ``(..., npix)`` for HEALPIX, or
        ``pixell.enmap.ndmap`` of shape ``(..., ny, nx)`` for CAR.

    Raises:
        ValueError: If both CAR and HEALPIX targets are specified, or if
            neither is.
    """
    car_target = out is not None or (shape is not None and wcs is not None)
    if car_target and nside is not None:
        raise ValueError("Specify either CAR (out / shape+wcs) or HEALPIX (nside), not both.")
    if car_target:
        if out is None:
            full_shape = (*alms.shape[:-1], *shape[-2:])
            out = enmap.zeros(full_shape, wcs=wcs, dtype=np.float64)
        return curvedsky.alm2map(alms, map=out, spin=spin, copy=True)
    if nside is None:
        raise ValueError("Provide nside for HEALPIX or out / shape+wcs for CAR.")
    work = alms[..., None, :] if spin == 0 else alms
    m = _alm2map_healpix(work, spin=spin, nside=nside, lmax=lmax, mmax=mmax)
    return m[..., 0, :] if spin == 0 else m


def synfast(cl, *, nside=None, shape=None, wcs=None, lmax=None, seed=None, new=True):
    """Generate a Gaussian random map from an input power spectrum.

    Args:
        cl: Power spectrum or list of spectra to sample from.
        nside: HEALPIX resolution. Mutually exclusive with ``shape``/``wcs``.
        shape: CAR pixel shape. Used together with ``wcs``.
        wcs: CAR world coordinate system. Used together with ``shape``.
        lmax: Bandlimit. Defaults to library default.
        seed: PRNG seed.
        new: HEALPIX-only: pass through to ``healpy.synfast``.

    Returns:
        ``np.ndarray`` for HEALPIX, ``pixell.enmap.ndmap`` for CAR.

    Raises:
        ValueError: If neither HEALPIX nor CAR targets are fully specified,
            or if both are.
    """
    car_target = shape is not None or wcs is not None
    if nside is not None and car_target:
        raise ValueError("Specify either nside (HEALPIX) or shape+wcs (CAR).")
    if nside is not None:
        if seed is not None:
            np.random.seed(seed)  # noqa: NPY002
        return hp.synfast(cl, nside=nside, lmax=lmax, new=new)
    if shape is None or wcs is None:
        raise ValueError("Provide nside, or both shape and wcs.")
    return curvedsky.rand_map(shape, wcs, np.atleast_2d(cl), lmax=lmax, seed=seed)


def almxfl(alms, fl, *, mmax=None, inplace=False):
    """Multiply each ``a_lm`` by ``f_l``.

    Pixel-agnostic since ducc0 and healpy share the triangular alm layout.

    Args:
        alms: Spherical harmonic coefficients, 1D for a single field or 2D
            ``(ncomp, nalm)`` for multiple components.
        fl: Multipole-dependent filter ``f_l`` of length ``lmax + 1``.
        mmax: Azimuthal bandlimit; defaults to inferred from ``alms``.
        inplace: If ``True``, modify ``alms`` directly.

    Returns:
        Filtered alms with the same shape as ``alms``.
    """
    if alms.ndim == 1:
        return hp.almxfl(alms, fl, mmax=mmax, inplace=inplace)
    out = alms if inplace else alms.copy()
    for i in range(out.shape[0]):
        hp.almxfl(out[i], fl, mmax=mmax, inplace=True)
    return out


def anafast(maps, maps2=None, *, lmax=None, mmax=None, niter=0, pol=True):
    """Compute auto or cross power spectrum.

    For HEALPIX inputs, delegates to ``healpy.anafast`` (which handles
    T-only and ``(T, Q, U) → (TT, EE, BB, TE, EB, TB)`` natively).
    For CAR inputs, uses ``pixell.curvedsky.map2alm`` then ``healpy.alm2cl``.

    Args:
        maps: Input map (HEALPIX ndarray or CAR enmap).
        maps2: Optional second map for cross-spectrum.
        lmax: Bandlimit.
        mmax: Azimuthal bandlimit (HEALPIX only).
        niter: Jacobi iterations for the forward SHT.
        pol: HEALPIX-only: pass through to ``healpy.anafast``.

    Returns:
        Power spectrum array. Shape depends on the number of input
        components and the underlying library.
    """
    if _is_car(maps):
        alm1 = curvedsky.map2alm(maps, lmax=lmax, niter=niter)
        alm2 = curvedsky.map2alm(maps2, lmax=lmax, niter=niter) if maps2 is not None else None
        return hp.alm2cl(alm1, alm2, lmax=lmax, mmax=mmax)
    return hp.anafast(maps, map2=maps2, lmax=lmax, mmax=mmax, iter=niter, pol=pol)
