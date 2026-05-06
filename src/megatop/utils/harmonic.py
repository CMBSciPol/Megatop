# Copyright (c) 2024–2026 members of the Simons Observatory Collaboration.
#
# This file is based on original SMARTIES code released under the MIT License.
# It has been modified from its original form.
#
# See the LICENSE file in the root of this repository for full MIT license terms.

"""Pixelization-agnostic spherical harmonic transforms.

Dispatches between HEALPIX (via ducc0) and CAR (via pixell.curvedsky) based
on the input map type. Public API takes keyword-only arguments and uses
native shapes for each pixelization: ``(..., npix)`` for HEALPIX and
``(..., ny, nx)`` for CAR.
"""

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


def _map2alm_healpix_iter(maps, *, spin, lmax=None, mmax=None, niter=3):
    nside = hp.npix2nside(maps.shape[-1])
    alm = _map2alm_healpix(maps, spin=spin, lmax=lmax, mmax=mmax)
    for _ in range(niter):
        residual = _alm2map_healpix(alm, spin=spin, nside=nside, lmax=lmax, mmax=mmax) - maps
        alm -= _map2alm_healpix(residual, spin=spin, lmax=lmax, mmax=mmax)
    return alm


def map2alm(maps, *, spin=0, lmax=None, mmax=None, niter=3):
    """Forward SHT, dispatching on pixelization.

    Args:
        maps: Input map. CAR shape ``(..., ny, nx)``; HEALPIX shape
            ``(..., npix)``. CAR inputs must be ``pixell.enmap.ndmap``.
        spin: Spin weight(s). Scalar (0 for T, 2 for Q/U) or a list such as
            ``[0, 2]`` to analyse mixed-spin fields from a stacked map array
            in one call.  CAR dispatches directly to pixell; HEALPIX splits
            the leading map axis by spin group.
        lmax: Bandlimit. Defaults to ``3 * nside - 1`` for HEALPIX and to
            the library default for CAR.
        mmax: Azimuthal bandlimit. Defaults to ``lmax``.
        niter: Iterative refinement steps. For HEALPIX, Jacobi iterations
            on top of adjoint synthesis. For CAR, passed through to
            ``pixell.curvedsky.map2alm``.

    Returns:
        Spherical harmonic coefficients with the last axis storing the
        triangular ``(l, m)`` layout.
    """
    if _is_car(maps):
        return curvedsky.map2alm(maps, spin=spin, lmax=lmax, niter=niter)
    if isinstance(spin, (list, tuple)):
        alms_out = []
        idx = 0
        for s in spin:
            nmaps = 1 if s == 0 else 2
            alms_out.append(
                _map2alm_healpix_iter(
                    maps[idx : idx + nmaps], spin=s, lmax=lmax, mmax=mmax, niter=niter
                )
            )
            idx += nmaps
        return np.concatenate(alms_out, axis=0)
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
        spin: Spin weight(s). Scalar (0 for T, 2 for Q/U) or a list such as
            ``[0, 2]`` to synthesise mixed-spin fields from a stacked ``alms``
            array in one call.  CAR dispatches directly to pixell; HEALPIX
            splits ``alms`` along the leading axis by spin group.
        nside: HEALPIX resolution. Mutually exclusive with the CAR options.
        shape: CAR pixel shape. Used together with ``wcs``.
        wcs: CAR world coordinate system. Used together with ``shape``.
        out: Pre-allocated CAR enmap written into in-place and returned.
            Mutually exclusive with ``nside``.
        lmax: Bandlimit. Defaults to inferred value from ``alms``.
        mmax: Azimuthal bandlimit. Defaults to ``lmax``.

    Returns:
        Pixel map. ``np.ndarray`` of shape ``(..., npix)`` for HEALPIX, or
        ``pixell.enmap.ndmap`` of shape ``(..., ny, nx)`` for CAR. When
        ``out`` is provided, the return value is ``out`` itself.

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
        return curvedsky.alm2map(alms, map=out, spin=spin, copy=False)
    if nside is None:
        raise ValueError("Provide nside for HEALPIX or out / shape+wcs for CAR.")
    if isinstance(spin, (list, tuple)):
        maps_out = []
        idx = 0
        for s in spin:
            nmaps = 1 if s == 0 else 2
            maps_out.append(
                _alm2map_healpix(alms[idx : idx + nmaps], spin=s, nside=nside, lmax=lmax, mmax=mmax)
            )
            idx += nmaps
        return np.concatenate(maps_out, axis=0)
    work = alms[..., None, :] if spin == 0 else alms
    m = _alm2map_healpix(work, spin=spin, nside=nside, lmax=lmax, mmax=mmax)
    return m[..., 0, :] if spin == 0 else m


def _normalise_cl(cl):
    """Normalise any supported cl format to a scalar array or list for ``hp.synalm``.

    * ``(lmax+1,)`` — returned as-is (single spectrum).
    * ``(nspec, lmax+1)`` flat diagonal ordering → list; caller controls
      ordering via the ``new`` argument to ``hp.synalm``.
    * ``(4, lmax+1)`` — healpy 4-spectrum shorthand ``TT EE BB TE``; padded
      to 6 spectra with EB=TB=0 in diagonal order.
    * ``(n, n, lmax+1)`` covariance matrix → upper triangle extracted in
      diagonal (``new=True``) order.
    """
    cl = np.asarray(cl)
    if cl.ndim == 1:
        return cl
    if cl.ndim == 3:
        n = cl.shape[0]
        out = []
        for diag in range(n):
            for i in range(n - diag):
                out.append(cl[i, i + diag])
        return out
    if cl.ndim != 2:
        raise ValueError(f"cl must be 1-, 2-, or 3-D, got shape {cl.shape}")
    nspec = cl.shape[0]
    if nspec == 4:
        padded = np.zeros((6, cl.shape[1]), dtype=cl.dtype)
        padded[:4] = cl
        return list(padded)
    n = int(round((-1 + np.sqrt(1 + 8 * nspec)) / 2))
    if n * (n + 1) // 2 != nspec:
        raise ValueError(
            f"cl has {nspec} spectra, which is not triangular (n*(n+1)/2). "
            "Pass an (n, n, lmax+1) covariance matrix or healpy-ordered flat spectra."
        )
    return list(cl)


def synfast(cl, *, nside=None, shape=None, wcs=None, lmax=None, seed=None, new=True):
    """Generate a Gaussian random map from an input power spectrum.

    Args:
        cl: Power spectrum. Accepted formats for both pixelizations:

            * 1-D ``(lmax+1,)`` — single TT spectrum.
            * 2-D ``(nspec, lmax+1)`` flat diagonal ordering
              ``TT, EE, BB, TE, EB, TB`` (or the 4-spectrum shorthand
              ``TT, EE, BB, TE`` with EB=TB=0).
            * 3-D ``(n, n, lmax+1)`` covariance matrix.

        nside: HEALPIX resolution. Mutually exclusive with ``shape``/``wcs``.
        shape: CAR pixel shape. Used together with ``wcs``.
        wcs: CAR world coordinate system. Used together with ``shape``.
        lmax: Bandlimit. Defaults to library default.
        seed: PRNG seed.
        new: Ordering convention for flat 2-D ``cl`` input, passed to
            ``healpy.synalm``. Defaults to ``True`` (diagonal ordering
            ``TT, EE, BB, TE, EB, TB``), which differs from healpy's own
            default of ``False``. Has no effect for 1-D or 3-D ``cl``.

    Returns:
        ``np.ndarray`` for HEALPIX, ``pixell.enmap.ndmap`` for CAR.

    Raises:
        ValueError: If neither HEALPIX nor CAR targets are fully specified,
            or if both are.
    """
    car_target = shape is not None or wcs is not None
    if nside is not None and car_target:
        raise ValueError("Specify either nside (HEALPIX) or shape+wcs (CAR).")
    if nside is None and (shape is None or wcs is None):
        raise ValueError("Provide nside, or both shape and wcs.")

    if seed is not None:
        np.random.seed(seed)  # noqa: NPY002

    cl_norm = _normalise_cl(cl)
    scalar = isinstance(cl_norm, np.ndarray) and cl_norm.ndim == 1

    if scalar:
        alm = hp.synalm(cl_norm, lmax=lmax, new=new)
        if nside is not None:
            return alm2map(alm, spin=0, nside=nside, lmax=lmax)
        return alm2map(alm, spin=0, shape=shape[-2:], wcs=wcs)

    # Multi-component (T, E, B) → synthesise (T, Q, U)
    alm_T, alm_E, alm_B = hp.synalm(cl_norm, lmax=lmax, new=new)
    alms_teb = np.stack([alm_T, alm_E, alm_B])
    if nside is not None:
        return alm2map(alms_teb, spin=[0, 2], nside=nside, lmax=lmax)
    return alm2map(alms_teb, spin=[0, 2], shape=shape[-2:], wcs=wcs)


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


def anafast(maps, maps2=None, *, lmax=None, mmax=None, niter=3, pol=True):
    """Compute auto or cross power spectrum.

    Both HEALPIX and CAR paths use ``map2alm`` (ducc0 / pixell) then
    ``healpy.alm2cl``.  For HEALPIX TQU maps (``pol=True``), T is
    decomposed with spin-0 and QU with spin-2 to obtain E/B alms, then
    ``healpy.alm2cl`` returns ``(TT, EE, BB, TE, EB, TB)``.

    Args:
        maps: Input map (HEALPIX ndarray or CAR enmap).
        maps2: Optional second map for cross-spectrum.
        lmax: Bandlimit.
        mmax: Azimuthal bandlimit (HEALPIX only).
        niter: Jacobi iterations for the forward SHT.
        pol: If ``True`` and HEALPIX input has shape ``(..., 3, npix)``,
            treat as TQU and return all six spectra.

    Returns:
        Power spectrum array. For TQU HEALPIX input (``pol=True``) the six
        spectra are returned in diagonal ordering ``TT, EE, BB, TE, EB, TB``
        — consistent with ``synfast``'s ``new=True`` convention and suitable
        as direct input to ``synfast``.
    """

    def _healpix_alms(m):
        if pol and m.ndim >= 2 and m.shape[-2] == 3:
            return map2alm(m, spin=[0, 2], lmax=lmax, mmax=mmax, niter=niter)
        return map2alm(m, spin=0, lmax=lmax, mmax=mmax, niter=niter)

    if _is_car(maps):
        alm1 = curvedsky.map2alm(maps, lmax=lmax, niter=niter)
        alm2 = curvedsky.map2alm(maps2, lmax=lmax, niter=niter) if maps2 is not None else None
    else:
        alm1 = _healpix_alms(maps)
        alm2 = _healpix_alms(maps2) if maps2 is not None else None
    return hp.alm2cl(alm1, alm2, lmax=lmax, mmax=mmax)
