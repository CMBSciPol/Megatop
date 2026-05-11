# Copyright (c) 2024–2026 members of the Simons Observatory Collaboration.
#
# This file is based on original SMARTIES code released under the MIT License.
# It has been modified from its original form.
#
# See the LICENSE file in the root of this repository for full MIT license terms.

"""Pixelization-agnostic spherical harmonic transforms.

Dispatches between HEALPix (via ducc0) and CAR (via pixell.curvedsky) based on
the input map type. Shapes are ``(..., npix)`` for HEALPix and ``(..., ny, nx)``
for CAR. All public functions are keyword-only.

Environment variables:
    MEGATOP_SHT_NTHREADS: Default ducc0 thread count (HEALPix path). Read once
        at import; ``0`` (default) lets ducc0 choose. Overridden per-call via
        ``nthreads`` (``0`` or ``None`` falls back to this default).
"""

import os
from functools import lru_cache

import ducc0
import healpy as hp
import numpy as np
from pixell import curvedsky, enmap

_DEFAULT_NTHREADS = int(os.environ.get("MEGATOP_SHT_NTHREADS", "0"))

__all__ = [
    "alm2map",
    "almxfl",
    "anafast",
    "getlmax",
    "map2alm",
    "gauss_beam",
    "smooth",
    "synfast",
]


def _is_car(x) -> bool:
    return isinstance(x, enmap.ndmap)


def getlmax(alm, mmax=None) -> int:
    """Infer ``lmax`` from ``alm.shape[-1]``.

    Args:
        alm: Alm array; last axis stores the ``(l, m)`` layout.
        mmax: Azimuthal bandlimit if the layout is truncated (``mmax < lmax``).
            Defaults to ``lmax`` (full triangular layout).

    Returns:
        ``lmax`` consistent with ``alm.shape[-1]`` and ``mmax``.
    """
    return hp.Alm.getlmax(alm.shape[-1], mmax=mmax)


@lru_cache(maxsize=8)
def _healpix_sht_info(nside):
    return ducc0.healpix.Healpix_Base(nside, "RING").sht_info()


def _ducc_sht_kwargs(*, spin, nside, lmax, mmax=None):
    if mmax is None:
        mmax = lmax
    return {"spin": spin, "lmax": lmax, "mmax": mmax, **_healpix_sht_info(nside)}


def _ducc_synthesis(alms, *, spin, nside, lmax=None, mmax=None, nthreads=None, out=None):
    """Thin ducc0 synthesis wrapper.

    ``alms`` must follow ducc0's ``([ntrans,] nmaps, nalm)`` convention:
    ``nmaps == 1`` for spin 0, ``nmaps == 2`` for spin > 0.
    ``out``, if provided, must match the expected ``([ntrans,] nmaps, npix)`` shape.
    """
    alm_lmax = getlmax(alms, mmax=mmax)
    if lmax is None:
        lmax = alm_lmax
    elif lmax > alm_lmax:
        raise ValueError(f"lmax={lmax} exceeds alm bandlimit {alm_lmax}")
    return ducc0.sht.synthesis(
        alm=alms,
        map=out,
        nthreads=nthreads or _DEFAULT_NTHREADS,
        **_ducc_sht_kwargs(spin=spin, nside=nside, lmax=lmax, mmax=mmax),
    )


def _alm2map_healpix(alms, *, spin, nside, lmax=None, mmax=None, nthreads=None, out=None):
    """Public-shape synthesis entrypoint.

    ``alms``: ``(..., nalm)`` for spin 0, ``(..., 2, nalm)`` for spin > 0.
    ``out``:  ``(..., npix)`` for spin 0, ``(..., 2, npix)`` for spin > 0.
    Injects and strips the ducc0 nmaps axis for spin 0 internally.
    """
    kw = {"nside": nside, "lmax": lmax, "mmax": mmax, "nthreads": nthreads}
    if spin != 0:
        # ducc0 writes into out (if given) and returns it, so this handles both cases
        return _ducc_synthesis(alms, spin=spin, out=out, **kw)
    # special case spin = 0
    inplace = out is not None
    ducc_out = out[..., None, :] if inplace else None
    m = _ducc_synthesis(alms[..., None, :], spin=0, out=ducc_out, **kw)
    return out if inplace else m[..., 0, :]


def _ducc_adjoint_synthesis(maps, *, spin, lmax=None, mmax=None, nthreads=None):
    """Thin ducc0 adjoint-synthesis wrapper.

    ``maps`` must follow ducc0's ``([ntrans,] nmaps, npix)`` convention.
    """
    nside = hp.npix2nside(maps.shape[-1])
    lmax_max = 3 * nside - 1
    if lmax is None:
        lmax = lmax_max
    elif lmax > lmax_max:
        raise ValueError(f"lmax={lmax} exceeds 3*nside-1={lmax_max}")
    weight = 4 * np.pi / (12 * nside**2)
    return ducc0.sht.adjoint_synthesis(
        map=maps * weight,
        nthreads=nthreads or _DEFAULT_NTHREADS,
        **_ducc_sht_kwargs(spin=spin, nside=nside, lmax=lmax, mmax=mmax),
    )


def _map2alm_healpix_iter(maps, *, spin, lmax=None, mmax=None, niter=3, nthreads=None):
    """Jacobi iteration over ``_ducc_adjoint_synthesis``. Uses ducc0 shapes."""
    nside = hp.npix2nside(maps.shape[-1])
    kw = {"spin": spin, "lmax": lmax, "mmax": mmax, "nthreads": nthreads}
    alm = _ducc_adjoint_synthesis(maps, **kw)
    for _ in range(niter):
        residual = _ducc_synthesis(alm, **kw, nside=nside) - maps
        alm -= _ducc_adjoint_synthesis(residual, **kw)
    return alm


def _map2alm_healpix(maps, *, spin, lmax=None, mmax=None, niter=3, nthreads=None):
    """Public-shape forward SHT entrypoint.

    ``maps``: ``(..., npix)`` for spin 0, ``(..., 2, npix)`` for spin > 0.
    Injects and strips the ducc0 nmaps axis for spin 0 internally.
    """
    kw = {"spin": spin, "lmax": lmax, "mmax": mmax, "niter": niter, "nthreads": nthreads}
    if spin == 0:
        alm = _map2alm_healpix_iter(maps[..., None, :], **kw)
        return alm[..., 0, :]
    return _map2alm_healpix_iter(maps, **kw)


def map2alm(maps, *, spin=0, lmax=None, mmax=None, niter=3, nthreads=None):
    """Forward SHT, dispatching on pixelization.

    Args:
        maps: HEALPix ``(..., npix)`` ndarray or CAR ``pixell.enmap.ndmap``
            ``(..., ny, nx)``.
        spin: Spin weight: ``0`` (T), ``2`` (Q/U), or a list like ``[0, 2]``
            for mixed-spin fields. HEALPix splits the map axis by spin group;
            CAR passes to pixell.

            Unlike healpy's ``pol=True``, TQU → TEB requires ``spin=[0, 2]``
            explicitly. With ``spin=[0, 2]`` and a ``(3, npix)`` input the
            output is ``(3, nalm)`` with rows ``[alm_T, alm_E, alm_B]``.
            Batch dimensions are supported: ``(batch, 3, npix)`` → ``(batch, 3, nalm)``.
        lmax: Bandlimit. HEALPix default: ``3 * nside - 1``; CAR: library default.
        mmax: Azimuthal bandlimit (HEALPix only). Defaults to ``lmax``.
        niter: Refinement steps. HEALPix: Jacobi iterations; CAR: passed to pixell.
        nthreads: ducc0 thread count (HEALPix). ``None`` uses ``MEGATOP_SHT_NTHREADS``.

    Returns:
        Alm array, last axis in triangular ``(l, m)`` layout.
    """
    if _is_car(maps):
        return curvedsky.map2alm(maps, spin=spin, lmax=lmax, niter=niter)
    kw = {"lmax": lmax, "mmax": mmax, "niter": niter, "nthreads": nthreads}
    if isinstance(spin, (list, tuple)):
        alms_out = []
        idx = 0
        for s in spin:
            nmaps = 1 if s == 0 else 2
            # _map2alm_healpix_iter uses ducc0 ([ntrans,] nmaps, npix) convention directly
            alms_out.append(_map2alm_healpix_iter(maps[..., idx : idx + nmaps, :], spin=s, **kw))
            idx += nmaps
        return np.concatenate(alms_out, axis=-2)
    return _map2alm_healpix(maps, spin=spin, **kw)


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
    nthreads=None,
):
    """Inverse SHT. Target pixelization set by kwargs:

    - ``out`` (enmap) → CAR in-place.
    - ``shape`` + ``wcs`` → CAR, new enmap.
    - ``nside`` → HEALPix.

    Args:
        alms: Spherical harmonic coefficients.
        spin: Spin weight: ``0`` (T), ``2`` (Q/U), or a list like ``[0, 2]``
            for mixed-spin fields. HEALPix splits the alm axis by spin group;
            CAR passes to pixell.

            TEB → TQU requires ``spin=[0, 2]``. With a ``(3, nalm)`` input the
            output is ``(3, npix)`` with rows ``[T, Q, U]``.
            Batch dimensions are supported: ``(batch, 3, nalm)`` → ``(batch, 3, npix)``.
        nside: HEALPix resolution. Mutually exclusive with CAR options.
        shape: CAR pixel shape. Used with ``wcs``.
        wcs: CAR world coordinate system. Used with ``shape``.
        out: Pre-allocated output written in-place and returned. enmap for CAR,
            ndarray for HEALPix. Mutually exclusive with ``shape``/``wcs``.
        lmax: Bandlimit (HEALPix only — CAR infers it from ``alms``).
        mmax: Azimuthal bandlimit (HEALPix only). Defaults to ``lmax``.
        nthreads: ducc0 thread count (HEALPix). ``None`` uses ``MEGATOP_SHT_NTHREADS``.

    Returns:
        ``np.ndarray`` ``(..., npix)`` for HEALPix or ``pixell.enmap.ndmap``
        ``(..., ny, nx)`` for CAR. Returns ``out`` when provided.

    Raises:
        ValueError: Conflicting or missing pixelization targets.
    """
    if out is not None and (shape is not None or wcs is not None):
        raise ValueError("Provide either out or shape+wcs, not both.")
    car_target = _is_car(out) or (shape is not None and wcs is not None)
    if car_target and nside is not None:
        raise ValueError("Specify either CAR (out / shape+wcs) or HEALPix (nside), not both.")
    if car_target:
        if out is None:
            full_shape = (*alms.shape[:-1], *shape[-2:])
            out = enmap.zeros(full_shape, wcs=wcs, dtype=alms.real.dtype)
        return curvedsky.alm2map(alms, map=out, spin=spin, copy=False)
    if nside is None:
        raise ValueError("Provide nside for HEALPix or out / shape+wcs for CAR.")
    kw = {"nside": nside, "lmax": lmax, "mmax": mmax, "nthreads": nthreads}
    if isinstance(spin, (list, tuple)):
        inplace = out is not None
        maps_out = [] if not inplace else None
        out_idx = idx = 0
        for s in spin:
            nmaps = 1 if s == 0 else 2
            # _ducc_synthesis uses ([ntrans,] nmaps, nalm) convention directly
            out_seg = out[..., out_idx : out_idx + nmaps, :] if inplace else None
            result = _ducc_synthesis(alms[..., idx : idx + nmaps, :], spin=s, out=out_seg, **kw)
            if not inplace:
                maps_out.append(result)
            idx += nmaps
            out_idx += nmaps
        return out if inplace else np.concatenate(maps_out, axis=-2)
    return _alm2map_healpix(alms, spin=spin, out=out, **kw)


def _normalise_cl(cl):
    """Normalise cl to a scalar array or list accepted by ``hp.synalm``.

    * ``(lmax+1,)`` — returned as-is.
    * ``(nspec, lmax+1)`` — returned as list; caller controls ordering via ``new``.
    * ``(4, lmax+1)`` — healpy shorthand TT EE BB TE (EB=TB=0 filled by healpy).
    * ``(n, n, lmax+1)`` — upper triangle extracted in diagonal (``new=True``) order.
    """
    cl = np.asarray(cl)
    if cl.ndim == 1:
        return cl
    if cl.ndim == 3:
        if cl.shape[0] != cl.shape[1]:
            raise ValueError(f"3-D cl must be square (n, n, lmax+1), got shape {cl.shape}")
        # extract upper triangle in "diagonal order" (TT EE BB TE EB TB)
        n = cl.shape[0]
        return [cl[i, i + d] for d in range(n) for i in range(n - d)]
    if cl.ndim != 2:
        raise ValueError(f"cl must be 1-, 2-, or 3-D, got shape {cl.shape}")
    nspec = cl.shape[0]
    if nspec == 4:
        return list(cl)
    n = int(round((-1 + np.sqrt(1 + 8 * nspec)) / 2))
    if n * (n + 1) // 2 != nspec:
        raise ValueError(
            f"cl has {nspec} spectra, which is not triangular (n*(n+1)/2). "
            "Pass an (n, n, lmax+1) covariance matrix or healpy-ordered flat spectra."
        )
    return list(cl)


def synfast(cl, *, nside=None, shape=None, wcs=None, lmax=None, seed=None, new=True, nthreads=None):
    """Generate a Gaussian random map from a power spectrum.

    Args:
        cl: Power spectrum. Accepted shapes:

            * 1-D ``(lmax+1,)`` — single TT spectrum.
            * 2-D ``(nspec, lmax+1)`` — flat spectra; ordering set by ``new``.
            * 2-D ``(4, lmax+1)`` — healpy shorthand TT EE BB TE (EB=TB=0).
            * 3-D ``(n, n, lmax+1)`` — covariance matrix.

        nside: HEALPix resolution. Mutually exclusive with ``shape``/``wcs``.
        shape: CAR pixel shape. Used with ``wcs``.
        wcs: CAR world coordinate system. Used with ``shape``.
        lmax: Bandlimit. Defaults to library default.
        seed: PRNG seed.
        nthreads: ducc0 thread count (HEALPix). ``None`` uses ``MEGATOP_SHT_NTHREADS``.
        new: Ordering for 2-D ``cl``, passed to ``hp.synalm``. Defaults to
            ``True`` (TT EE BB TE EB TB), unlike healpy's default ``False``.
            No effect for 1-D or 3-D input.

    Returns:
        ``np.ndarray`` for HEALPix, ``pixell.enmap.ndmap`` for CAR.

    Raises:
        ValueError: Missing or conflicting pixelization targets.
    """
    car_target = shape is not None or wcs is not None
    healpix = nside is not None
    if healpix and car_target:
        raise ValueError("Specify either nside (HEALPix) or shape+wcs (CAR).")
    if not healpix and (shape is None or wcs is None):
        raise ValueError("Provide nside, or both shape and wcs.")

    if seed is not None:
        np.random.seed(seed)  # noqa: NPY002

    cl = np.asarray(cl)
    if cl.ndim == 3:
        new = True
    cl_norm = _normalise_cl(cl)
    scalar = isinstance(cl_norm, np.ndarray) and cl_norm.ndim == 1
    kw = (
        {"nside": nside, "lmax": lmax, "nthreads": nthreads}
        if healpix
        else {"shape": shape[-2:], "wcs": wcs, "lmax": lmax}
    )

    if scalar:
        alm = hp.synalm(cl_norm, lmax=lmax, new=new)
        return alm2map(alm, spin=0, **kw)

    # Multi-component (T, E, B) → synthesise (T, Q, U)
    alm_T, alm_E, alm_B = hp.synalm(cl_norm, lmax=lmax, new=new)
    alms_teb = np.stack([alm_T, alm_E, alm_B])
    return alm2map(alms_teb, spin=[0, 2], **kw)


def almxfl(alms, fl, *, mmax=None, inplace=False):
    """Multiply each ``a_lm`` by ``f_l``. Pixel-agnostic.

    Args:
        alms: 1-D alm array or 2-D ``(ncomp, nalm)``.
        fl: Filter of length ``lmax + 1``.
        mmax: Azimuthal bandlimit. Defaults to ``lmax`` (full triangular layout).
        inplace: Modify ``alms`` in place.

    Returns:
        Filtered alms, same shape as input.
    """
    if alms.ndim == 1:
        return hp.almxfl(alms, fl, mmax=mmax, inplace=inplace)
    out = alms if inplace else alms.copy()
    for i in range(out.shape[0]):
        hp.almxfl(out[i], fl, mmax=mmax, inplace=True)
    return out


def gauss_beam(fwhm_arcmin, lmax, *, pol=False):
    """Wrapper around ``healpy.gauss_beam`` with FWHM in arcminutes.

    See healpy documentation for full details. The only difference is that
    ``fwhm_arcmin`` is in arcminutes whereas healpy's ``fwhm`` is in radians.
    """
    return hp.gauss_beam(np.radians(fwhm_arcmin / 60), lmax=lmax, pol=pol)


def smooth(
    maps,
    fwhm_arcmin,
    *,
    pol=False,
    lmax=None,
    nside=None,
    shape=None,
    wcs=None,
    out=None,
    niter=3,
    nthreads=None,
):
    """Smooth a map with a Gaussian beam.

    When no output geometry is given the input geometry is reused:
    HEALPix nside is inferred from the input pixel count; CAR shape and
    WCS are copied from the input enmap.

    Args:
        maps: Input map — HEALPix ``(..., npix)`` ndarray or CAR
            ``pixell.enmap.ndmap`` ``(..., ny, nx)``.
        fwhm_arcmin: FWHM of the Gaussian beam in arcminutes.
        pol: If ``True``, treat ``maps`` as TQU and apply separate T and P
            beams (spin ``[0, 2]``). If ``False`` (default), apply a single
            spin-0 beam to all components.
        lmax: Bandlimit. Inferred from the alm output of ``map2alm`` if
            ``None``.
        nside: HEALPix output resolution. Defaults to input nside.
        shape: CAR output pixel shape. Defaults to input shape.
        wcs: CAR world coordinate system. Defaults to input WCS.
        out: Pre-allocated output map written in-place and returned.
        niter: Jacobi iterations for the forward SHT (HEALPix).
        nthreads: ducc0 thread count (HEALPix).

    Returns:
        Smoothed map, same type and geometry as input unless overridden.
    """
    spin = [0, 2] if pol else 0
    alms = map2alm(maps, spin=spin, lmax=lmax, niter=niter, nthreads=nthreads)
    lmax_alm = getlmax(alms)
    if pol:
        bl = gauss_beam(fwhm_arcmin, lmax_alm, pol=True)  # (lmax+1, 4)
        almxfl(alms[0], bl[:, 0], inplace=True)
        almxfl(alms[1:], bl[:, 1], inplace=True)
    else:
        almxfl(alms, gauss_beam(fwhm_arcmin, lmax_alm), inplace=True)
    if _is_car(maps):
        if out is None and shape is None:
            shape = maps.shape
            wcs = maps.wcs
    elif out is None and nside is None and shape is None:
        nside = hp.npix2nside(np.asarray(maps).shape[-1])
    return alm2map(alms, spin=spin, nside=nside, shape=shape, wcs=wcs, out=out, nthreads=nthreads)


def anafast(maps, maps2=None, *, lmax=None, mmax=None, niter=3, pol=True, nthreads=None):
    """Compute auto or cross power spectrum.

    Routes through ``map2alm`` then ``hp.alm2cl``. For TQU input (``pol=True``),
    spin-0/spin-2 decomposition yields TEB alms; output is six spectra
    TT EE BB TE EB TB in diagonal order, consistent with ``synfast(new=True)``.

    Args:
        maps: Input map (HEALPix ndarray or CAR enmap).
        maps2: Second map for cross-spectrum.
        lmax: Bandlimit.
        mmax: Azimuthal bandlimit (HEALPix only — pixell does not expose it).
        niter: Jacobi iterations for the forward SHT.
        pol: If ``True`` and the map has a Stokes axis of length 3, decompose
            into TEB and return all six spectra. Raises ``ValueError`` if the
            Stokes axis exists but has length ≠ 3.
        nthreads: ducc0 thread count (HEALPix only).

    Returns:
        Power spectrum array. For TQU input: TT EE BB TE EB TB in diagonal
        order, directly usable as ``synfast`` input.
    """

    def _alms(m):
        if m is None:
            return None
        stokes_axis = -3 if _is_car(m) else -2
        has_stokes = m.ndim >= abs(stokes_axis)
        if pol and has_stokes and m.shape[stokes_axis] != 3:
            raise ValueError(
                f"pol=True requires 3 Stokes components along axis {stokes_axis}, "
                f"got shape {m.shape}"
            )
        is_tqu = pol and has_stokes and m.shape[stokes_axis] == 3
        spin = [0, 2] if is_tqu else 0
        return map2alm(m, spin=spin, lmax=lmax, mmax=mmax, niter=niter, nthreads=nthreads)

    return hp.alm2cl(_alms(maps), _alms(maps2), lmax=lmax, mmax=mmax)
