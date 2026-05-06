"""Tests for the dispatch and convenience layer in megatop.utils.harmonic.

Underlying SHT correctness is the responsibility of ducc0 / pixell / healpy
and is not retested here. These tests only exercise behavior the wrapper
itself adds: pixelization dispatch, spin-0 batching, 2D almxfl, anafast
routing, argument validation.
"""

import healpy as hp
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

pytest.importorskip("ducc0")
pytest.importorskip("pixell")

from pixell import enmap

from megatop.utils import harmonic

NSIDE = 8
LMAX = 2 * NSIDE
NALM = hp.Alm.getsize(LMAX)
RNG = np.random.default_rng(0)


@pytest.fixture(scope="module")
def car_geometry():
    return enmap.fullsky_geometry(res=np.deg2rad(10.0))


@pytest.fixture(scope="module")
def car_tqu_geometry(car_geometry):
    shape, wcs = car_geometry
    return (3, *shape[-2:]), wcs


def _random_alm(ncomp=1):
    shape = (ncomp, NALM) if ncomp > 1 else (NALM,)
    return RNG.standard_normal(shape) + 1j * RNG.standard_normal(shape)


# --- dispatch ---------------------------------------------------------------


def test_map2alm_dispatch_healpix():
    m = RNG.standard_normal(hp.nside2npix(NSIDE))
    alm = harmonic.map2alm(m, spin=0, lmax=LMAX)
    assert isinstance(alm, np.ndarray)
    assert not isinstance(alm, enmap.ndmap)


def test_map2alm_dispatch_car(car_geometry):
    shape, wcs = car_geometry
    m = enmap.enmap(RNG.standard_normal(shape), wcs)
    alm = harmonic.map2alm(m, spin=0, lmax=LMAX)
    assert isinstance(alm, np.ndarray)


def test_alm2map_dispatch_healpix():
    alm = _random_alm()
    m = harmonic.alm2map(alm, spin=0, nside=NSIDE, lmax=LMAX)
    assert m.shape[-1] == hp.nside2npix(NSIDE)
    assert not isinstance(m, enmap.ndmap)


def test_alm2map_dispatch_car_via_shape_wcs(car_geometry):
    shape, wcs = car_geometry
    alm = _random_alm()
    m = harmonic.alm2map(alm, spin=0, shape=shape, wcs=wcs, lmax=LMAX)
    assert isinstance(m, enmap.ndmap)
    assert m.shape[-2:] == tuple(shape[-2:])


def test_alm2map_dispatch_car_via_out(car_geometry):
    shape, wcs = car_geometry
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


def test_anafast_healpix_scalar():
    m = RNG.standard_normal(hp.nside2npix(NSIDE))
    cl_us = harmonic.anafast(m, lmax=LMAX, pol=False, niter=3)
    cl_hp = hp.anafast(m, lmax=LMAX, pol=False, iter=3)
    assert_allclose(cl_us, cl_hp, rtol=1e-10)


def test_anafast_healpix_pol():
    tqu = RNG.standard_normal((3, hp.nside2npix(NSIDE)))
    cl_us = harmonic.anafast(tqu, lmax=LMAX, pol=True, niter=3)
    cl_hp = hp.anafast(tqu, lmax=LMAX, pol=True, iter=3)
    assert cl_us.shape == cl_hp.shape  # (6, lmax+1)
    assert_allclose(cl_us, cl_hp, rtol=1e-10)


def test_anafast_healpix_cross():
    npix = hp.nside2npix(NSIDE)
    m1 = RNG.standard_normal(npix)
    m2 = RNG.standard_normal(npix)
    cl_us = harmonic.anafast(m1, m2, lmax=LMAX, pol=False, niter=3)
    cl_hp = hp.anafast(m1, map2=m2, lmax=LMAX, pol=False, iter=3)
    assert_allclose(cl_us, cl_hp, rtol=1e-10)


def test_anafast_car_auto(car_geometry):
    shape, wcs = car_geometry
    m = enmap.enmap(RNG.standard_normal(shape), wcs)
    cl = harmonic.anafast(m, lmax=LMAX)
    assert cl.shape[-1] == LMAX + 1


def test_anafast_car_cross(car_geometry):
    shape, wcs = car_geometry
    m1 = enmap.enmap(RNG.standard_normal(shape), wcs)
    m2 = enmap.enmap(RNG.standard_normal(shape), wcs)
    cl = harmonic.anafast(m1, m2, lmax=LMAX)
    assert cl.shape[-1] == LMAX + 1


# --- argument validation ---------------------------------------------------


def test_alm2map_requires_target():
    alm = _random_alm()
    with pytest.raises(ValueError):
        harmonic.alm2map(alm, spin=0)


def test_alm2map_rejects_both_targets(car_geometry):
    shape, wcs = car_geometry
    alm = _random_alm()
    with pytest.raises(ValueError):
        harmonic.alm2map(alm, spin=0, nside=NSIDE, shape=shape, wcs=wcs)


def test_synfast_rejects_both_targets(car_geometry):
    shape, wcs = car_geometry
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


def _make_cls(lmax):
    """CMB-like spectra: power-law TT with non-zero TE, zero EB/TB."""
    ell = np.arange(lmax + 1, dtype=float)
    ell[0] = 1  # avoid divide-by-zero; TT[0] set to 0 below
    tt = 1e-4 / ell**2
    tt[0] = 0.0
    ee = tt * 0.05
    bb = tt * 0.01
    te = 0.3 * np.sqrt(tt * ee)  # |TE|^2 = 0.09 * TT*EE < TT*EE → positive-definite
    cls = np.zeros((6, lmax + 1))
    cls[0] = tt
    cls[1] = ee
    cls[2] = bb
    cls[3] = te  # TE non-zero to exercise cross-correlation path
    return cls


_CL_T = _make_cls(LMAX_SF)[0]
_CL_POL = _make_cls(LMAX_SF)


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


# --- _flat_cl_to_cov --------------------------------------------------------


NL = LMAX + 1


def test_flat_cl_to_cov_1d():
    cl = np.ones(NL)
    cov = harmonic._flat_cl_to_cov(cl)
    assert cov.shape == (1, 1, NL)
    assert_array_equal(cov[0, 0], cl)


def test_flat_cl_to_cov_3d_passthrough():
    cov_in = np.ones((3, 3, NL))
    cov_out = harmonic._flat_cl_to_cov(cov_in)
    assert cov_out is cov_in


def test_flat_cl_to_cov_n2_triangular():
    # TT, EE, TE (diagonal ordering, n=2)
    cl = np.arange(3 * NL, dtype=float).reshape(3, NL)
    cov = harmonic._flat_cl_to_cov(cl)
    assert cov.shape == (2, 2, NL)
    assert_array_equal(cov[0, 0], cl[0])  # TT
    assert_array_equal(cov[1, 1], cl[1])  # EE
    assert_array_equal(cov[0, 1], cl[2])  # TE
    assert_array_equal(cov[1, 0], cl[2])  # TE symmetric


def test_flat_cl_to_cov_n3_triangular():
    # TT, EE, BB, TE, EB, TB (diagonal ordering, n=3)
    cl = np.arange(6 * NL, dtype=float).reshape(6, NL)
    cov = harmonic._flat_cl_to_cov(cl)
    assert cov.shape == (3, 3, NL)
    assert_array_equal(cov[0, 0], cl[0])  # TT
    assert_array_equal(cov[1, 1], cl[1])  # EE
    assert_array_equal(cov[2, 2], cl[2])  # BB
    assert_array_equal(cov[0, 1], cl[3])  # TE
    assert_array_equal(cov[1, 0], cl[3])  # TE symmetric
    assert_array_equal(cov[1, 2], cl[4])  # EB
    assert_array_equal(cov[2, 1], cl[4])  # EB symmetric
    assert_array_equal(cov[0, 2], cl[5])  # TB
    assert_array_equal(cov[2, 0], cl[5])  # TB symmetric


def test_flat_cl_to_cov_4spec_healpy_shorthand():
    # TT, EE, BB, TE — EB and TB assumed zero
    cl = np.arange(4 * NL, dtype=float).reshape(4, NL)
    cov = harmonic._flat_cl_to_cov(cl)
    assert cov.shape == (3, 3, NL)
    assert_array_equal(cov[0, 0], cl[0])  # TT
    assert_array_equal(cov[1, 1], cl[1])  # EE
    assert_array_equal(cov[2, 2], cl[2])  # BB
    assert_array_equal(cov[0, 1], cl[3])  # TE
    assert_array_equal(cov[1, 0], cl[3])  # TE symmetric
    assert_array_equal(cov[1, 2], 0)  # EB = 0
    assert_array_equal(cov[0, 2], 0)  # TB = 0


def test_flat_cl_to_cov_invalid_nspec_raises():
    with pytest.raises(ValueError, match="triangular"):
        harmonic._flat_cl_to_cov(np.ones((5, NL)))


# --- synfast CAR (cl format support) ----------------------------------------


LMAX_CAR = 20

_CL_6_CAR = _make_cls(LMAX_CAR)
_CL_TT_CAR = _CL_6_CAR[0]


def test_synfast_car_1d_shape(car_geometry):
    shape, wcs = car_geometry
    m = harmonic.synfast(_CL_TT_CAR, shape=shape, wcs=wcs, lmax=LMAX_CAR, seed=1)
    assert isinstance(m, enmap.ndmap)
    assert np.all(np.isfinite(m))


def test_synfast_car_flat_6spec_shape(car_tqu_geometry):
    tqu_shape, wcs = car_tqu_geometry
    m = harmonic.synfast(_CL_6_CAR, shape=tqu_shape, wcs=wcs, lmax=LMAX_CAR, seed=2)
    assert isinstance(m, enmap.ndmap)
    assert m.shape[-3] == 3
    assert np.all(np.isfinite(m))


def test_synfast_car_4spec_shape(car_tqu_geometry):
    tqu_shape, wcs = car_tqu_geometry
    m = harmonic.synfast(_CL_6_CAR[:4], shape=tqu_shape, wcs=wcs, lmax=LMAX_CAR, seed=3)
    assert isinstance(m, enmap.ndmap)
    assert m.shape[-3] == 3
    assert np.all(np.isfinite(m))


def test_synfast_car_cov_matrix_shape(car_tqu_geometry):
    tqu_shape, wcs = car_tqu_geometry
    cov = harmonic._flat_cl_to_cov(_CL_6_CAR)
    m = harmonic.synfast(cov, shape=tqu_shape, wcs=wcs, lmax=LMAX_CAR, seed=4)
    assert isinstance(m, enmap.ndmap)
    assert m.shape[-3] == 3
    assert np.all(np.isfinite(m))


def test_synfast_car_flat_and_cov_equivalent(car_tqu_geometry):
    """Flat 6-spectrum and covariance-matrix formats give identical results."""
    tqu_shape, wcs = car_tqu_geometry
    cov = harmonic._flat_cl_to_cov(_CL_6_CAR)
    m_flat = harmonic.synfast(_CL_6_CAR, shape=tqu_shape, wcs=wcs, lmax=LMAX_CAR, seed=7)
    m_cov = harmonic.synfast(cov, shape=tqu_shape, wcs=wcs, lmax=LMAX_CAR, seed=7)
    assert_array_equal(m_flat, m_cov)


def test_synfast_car_invalid_flat_raises(car_geometry):
    shape, wcs = car_geometry
    with pytest.raises(ValueError, match="triangular"):
        harmonic.synfast(np.ones((5, LMAX_CAR + 1)), shape=shape, wcs=wcs, lmax=LMAX_CAR)


# --- getlmax convenience ---------------------------------------------------


def test_getlmax_1d():
    fake = np.zeros(NALM, dtype=complex)
    assert harmonic.getlmax(fake) == LMAX


def test_getlmax_2d_uses_last_axis():
    fake = np.zeros((3, NALM), dtype=complex)
    assert harmonic.getlmax(fake) == LMAX
