"""Tests for the dispatch and convenience layer in megatop.utils.harmonic.

Underlying SHT correctness is the responsibility of ducc0 / pixell / healpy
and is not retested here. These tests only exercise behavior the wrapper
itself adds: pixelization dispatch, spin-0 batching, 2D almxfl, anafast
routing, argument validation.
"""

import healpy as hp
import numpy as np
import pytest

pytest.importorskip("ducc0")
pytest.importorskip("pixell")

from pixell import enmap

from megatop.utils import harmonic

NSIDE = 8
LMAX = 2 * NSIDE
NALM = hp.Alm.getsize(LMAX)
RNG = np.random.default_rng(0)


def _car_geometry():
    return enmap.fullsky_geometry(res=np.deg2rad(10.0))


def _random_alm(ncomp=1):
    shape = (ncomp, NALM) if ncomp > 1 else (NALM,)
    return RNG.standard_normal(shape) + 1j * RNG.standard_normal(shape)


# --- dispatch ---------------------------------------------------------------


def test_map2alm_dispatch_healpix():
    m = RNG.standard_normal(hp.nside2npix(NSIDE))
    alm = harmonic.map2alm(m, spin=0, lmax=LMAX)
    assert isinstance(alm, np.ndarray)
    assert not isinstance(alm, enmap.ndmap)


def test_map2alm_dispatch_car():
    shape, wcs = _car_geometry()
    m = enmap.enmap(RNG.standard_normal(shape), wcs)
    alm = harmonic.map2alm(m, spin=0, lmax=LMAX)
    assert isinstance(alm, np.ndarray)


def test_alm2map_dispatch_healpix():
    alm = _random_alm()
    m = harmonic.alm2map(alm, spin=0, nside=NSIDE, lmax=LMAX)
    assert m.shape[-1] == hp.nside2npix(NSIDE)
    assert not isinstance(m, enmap.ndmap)


def test_alm2map_dispatch_car_via_shape_wcs():
    shape, wcs = _car_geometry()
    alm = _random_alm()
    m = harmonic.alm2map(alm, spin=0, shape=shape, wcs=wcs, lmax=LMAX)
    assert isinstance(m, enmap.ndmap)
    assert m.shape[-2:] == tuple(shape[-2:])


def test_alm2map_dispatch_car_via_out():
    shape, wcs = _car_geometry()
    out = enmap.zeros(shape, wcs=wcs)
    alm = _random_alm()
    m = harmonic.alm2map(alm, spin=0, out=out, lmax=LMAX)
    assert isinstance(m, enmap.ndmap)
    assert m is out  # out written in-place, return value is the same object
    assert not np.all(out == 0)  # actually populated


# --- spin-0 batching (our addition over ducc0) -----------------------------


def test_map2alm_spin0_batches_leading_dim():
    ncomp = 4
    m = RNG.standard_normal((ncomp, hp.nside2npix(NSIDE)))
    alm = harmonic.map2alm(m, spin=0, lmax=LMAX)
    assert alm.shape == (ncomp, NALM)
    # equivalent to per-row calls
    for i in range(ncomp):
        alm_i = harmonic.map2alm(m[i], spin=0, lmax=LMAX)
        assert np.allclose(alm[i], alm_i)


def test_alm2map_spin0_batches_leading_dim():
    ncomp = 4
    alms = _random_alm(ncomp=ncomp)
    m = harmonic.alm2map(alms, spin=0, nside=NSIDE, lmax=LMAX)
    assert m.shape == (ncomp, hp.nside2npix(NSIDE))
    for i in range(ncomp):
        m_i = harmonic.alm2map(alms[i], spin=0, nside=NSIDE, lmax=LMAX)
        assert np.allclose(m[i], m_i)


# --- almxfl wrapper --------------------------------------------------------


def test_almxfl_1d_passthrough():
    alm = _random_alm()
    fl = np.linspace(1.0, 2.0, LMAX + 1)
    out = harmonic.almxfl(alm, fl)
    assert np.allclose(out, hp.almxfl(alm.copy(), fl))


def test_almxfl_2d_per_row():
    alms = _random_alm(ncomp=3)
    fl = np.linspace(1.0, 2.0, LMAX + 1)
    out = harmonic.almxfl(alms, fl)
    assert out.shape == alms.shape
    for i in range(3):
        assert np.allclose(out[i], hp.almxfl(alms[i].copy(), fl))


def test_almxfl_inplace():
    alms = _random_alm(ncomp=2)
    snapshot = alms.copy()
    fl = np.linspace(1.0, 2.0, LMAX + 1)
    out = harmonic.almxfl(alms, fl, inplace=True)
    assert out is alms
    for i in range(2):
        assert np.allclose(alms[i], hp.almxfl(snapshot[i].copy(), fl))


def test_almxfl_no_inplace_does_not_mutate():
    alms = _random_alm(ncomp=2)
    snapshot = alms.copy()
    fl = np.linspace(1.0, 2.0, LMAX + 1)
    harmonic.almxfl(alms, fl, inplace=False)
    assert np.allclose(alms, snapshot)


# --- anafast wrapper -------------------------------------------------------


def test_anafast_healpix_delegates_to_healpy():
    m = RNG.standard_normal(hp.nside2npix(NSIDE))
    cl_us = harmonic.anafast(m, lmax=LMAX, pol=False, niter=3)
    cl_hp = hp.anafast(m, lmax=LMAX, pol=False, iter=3)
    assert np.array_equal(cl_us, cl_hp)


def test_anafast_car_auto():
    shape, wcs = _car_geometry()
    m = enmap.enmap(RNG.standard_normal(shape), wcs)
    cl = harmonic.anafast(m, lmax=LMAX)
    assert cl.shape[-1] == LMAX + 1


def test_anafast_car_cross():
    shape, wcs = _car_geometry()
    m1 = enmap.enmap(RNG.standard_normal(shape), wcs)
    m2 = enmap.enmap(RNG.standard_normal(shape), wcs)
    cl = harmonic.anafast(m1, m2, lmax=LMAX)
    assert cl.shape[-1] == LMAX + 1


# --- argument validation ---------------------------------------------------


def test_alm2map_requires_target():
    alm = _random_alm()
    with pytest.raises(ValueError):
        harmonic.alm2map(alm, spin=0)


def test_alm2map_rejects_both_targets():
    shape, wcs = _car_geometry()
    alm = _random_alm()
    with pytest.raises(ValueError):
        harmonic.alm2map(alm, spin=0, nside=NSIDE, shape=shape, wcs=wcs)


def test_synfast_rejects_both_targets():
    shape, wcs = _car_geometry()
    with pytest.raises(ValueError):
        harmonic.synfast(np.ones(LMAX + 1), nside=NSIDE, shape=shape, wcs=wcs)


def test_synfast_requires_target():
    with pytest.raises(ValueError):
        harmonic.synfast(np.ones(LMAX + 1))


def test_alm2map_lmax_exceeds_alm_bandlimit():
    alm = _random_alm()
    with pytest.raises(ValueError, match="lmax"):
        harmonic.alm2map(alm, spin=0, nside=NSIDE, lmax=LMAX + 5)


# --- synfast healpix -------------------------------------------------------


NSIDE_SF = 64
LMAX_SF = 2 * NSIDE_SF
NPIX_SF = hp.nside2npix(NSIDE_SF)

# Flat-ish spectrum well above noise for statistical checks.
_CL_T = np.ones(LMAX_SF + 1) * 1e-4
_CL_POL = np.zeros((6, LMAX_SF + 1))
_CL_POL[0] = _CL_T  # TT
_CL_POL[1] = _CL_T * 0.5  # EE
_CL_POL[2] = _CL_T * 0.1  # BB


def test_synfast_healpix_t_shape():
    m = harmonic.synfast(_CL_T, nside=NSIDE_SF, lmax=LMAX_SF, seed=1)
    assert m.shape == (NPIX_SF,)
    assert m.dtype.kind == "f"
    assert np.all(np.isfinite(m))


def test_synfast_healpix_pol_shape():
    m = harmonic.synfast(_CL_POL, nside=NSIDE_SF, lmax=LMAX_SF, seed=2)
    assert m.shape == (3, NPIX_SF)
    assert m.dtype.kind == "f"
    assert np.all(np.isfinite(m))


def test_synfast_healpix_t_seed_reproducible():
    m1 = harmonic.synfast(_CL_T, nside=NSIDE_SF, lmax=LMAX_SF, seed=42)
    np.random.seed(None)  # noqa: NPY002
    m2 = harmonic.synfast(_CL_T, nside=NSIDE_SF, lmax=LMAX_SF, seed=42)
    assert np.array_equal(m1, m2)


def test_synfast_healpix_pol_seed_reproducible():
    m1 = harmonic.synfast(_CL_POL, nside=NSIDE_SF, lmax=LMAX_SF, seed=42)
    np.random.seed(None)  # noqa: NPY002
    m2 = harmonic.synfast(_CL_POL, nside=NSIDE_SF, lmax=LMAX_SF, seed=42)
    assert np.array_equal(m1, m2)


def test_synfast_healpix_t_matches_hp_synfast():
    np.random.seed(99)  # noqa: NPY002
    expected = hp.synfast(_CL_T, nside=NSIDE_SF, lmax=LMAX_SF, new=True)
    result = harmonic.synfast(_CL_T, nside=NSIDE_SF, lmax=LMAX_SF, seed=99)
    assert np.allclose(result, expected, rtol=1e-10)


def test_synfast_healpix_pol_matches_hp_synfast():
    np.random.seed(99)  # noqa: NPY002
    expected = hp.synfast(_CL_POL, nside=NSIDE_SF, lmax=LMAX_SF, new=True)
    result = harmonic.synfast(_CL_POL, nside=NSIDE_SF, lmax=LMAX_SF, seed=99)
    assert np.allclose(result, expected, rtol=1e-10)


# --- getlmax convenience ---------------------------------------------------


def test_getlmax_1d():
    fake = np.zeros(NALM, dtype=complex)
    assert harmonic.getlmax(fake) == LMAX


def test_getlmax_2d_uses_last_axis():
    fake = np.zeros((3, NALM), dtype=complex)
    assert harmonic.getlmax(fake) == LMAX
