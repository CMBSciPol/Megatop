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


# ---------------------------------------------------------------------------
# map2alm
# ---------------------------------------------------------------------------


class TestMap2Alm:
    def test_dispatch_healpix(self):
        m = RNG.standard_normal(hp.nside2npix(NSIDE))
        alm = harmonic.map2alm(m, spin=0, lmax=LMAX)
        assert isinstance(alm, np.ndarray)
        assert not isinstance(alm, enmap.ndmap)

    def test_dispatch_car(self, car_geometry):
        shape, wcs = car_geometry
        m = enmap.enmap(RNG.standard_normal(shape), wcs)
        alm = harmonic.map2alm(m, spin=0, lmax=LMAX)
        assert isinstance(alm, np.ndarray)

    def test_healpix_spin0_scalar(self):
        m = RNG.standard_normal(hp.nside2npix(NSIDE))
        alm = harmonic.map2alm(m, spin=0, lmax=LMAX)
        assert alm.shape == (NALM,)

    def test_healpix_spin0_batch(self):
        ncomp = 4
        m = RNG.standard_normal((ncomp, hp.nside2npix(NSIDE)))
        alm = harmonic.map2alm(m, spin=0, lmax=LMAX)
        assert alm.shape == (ncomp, NALM)
        for i in range(ncomp):
            assert_allclose(alm[i], harmonic.map2alm(m[i], spin=0, lmax=LMAX))

    def test_healpix_spin2(self):
        m = RNG.standard_normal((2, hp.nside2npix(NSIDE)))
        alm = harmonic.map2alm(m, spin=2, lmax=LMAX)
        assert alm.shape == (2, NALM)
        assert_allclose(alm, harmonic.map2alm(m, spin=[2], lmax=LMAX))

    def test_healpix_list_spin(self):
        npix = hp.nside2npix(NSIDE)
        map_T = RNG.standard_normal(npix)
        map_QU = RNG.standard_normal((2, npix))
        maps_tqu = np.concatenate([map_T[None], map_QU], axis=0)
        alms = harmonic.map2alm(maps_tqu, spin=[0, 2], lmax=LMAX)
        assert alms.shape == (3, NALM)
        assert_allclose(alms[0], harmonic.map2alm(map_T, spin=0, lmax=LMAX))
        assert_allclose(alms[1:], harmonic.map2alm(map_QU, spin=2, lmax=LMAX))

    def test_healpix_niter0_differs_from_niter3(self):
        m = RNG.standard_normal(hp.nside2npix(NSIDE))
        alm0 = harmonic.map2alm(m, spin=0, lmax=LMAX, niter=0)
        alm3 = harmonic.map2alm(m, spin=0, lmax=LMAX, niter=3)
        assert alm0.shape == alm3.shape
        assert not np.allclose(alm0, alm3)

    def test_healpix_lmax_exceeds_nside_limit(self):
        m = RNG.standard_normal(hp.nside2npix(NSIDE))
        with pytest.raises(ValueError, match="lmax"):
            harmonic.map2alm(m, spin=0, lmax=3 * NSIDE)

    def test_healpix_nthreads(self):
        m = RNG.standard_normal(hp.nside2npix(NSIDE))
        alm1 = harmonic.map2alm(m, spin=0, lmax=LMAX, nthreads=1)
        alm2 = harmonic.map2alm(m, spin=0, lmax=LMAX, nthreads=2)
        assert_allclose(alm1, alm2, rtol=1e-10)

    def test_car_list_spin(self, car_geometry):
        shape, wcs = car_geometry
        npix = shape[-2:]
        tqu = enmap.enmap(RNG.standard_normal((3, *npix)), wcs)
        alms = harmonic.map2alm(tqu, spin=[0, 2], lmax=LMAX)
        assert alms.shape == (3, NALM)
        assert_allclose(alms[0], harmonic.map2alm(tqu[0], spin=0, lmax=LMAX))
        assert_allclose(alms[1:], harmonic.map2alm(tqu[1:], spin=2, lmax=LMAX))


# ---------------------------------------------------------------------------
# alm2map
# ---------------------------------------------------------------------------


class TestAlm2Map:
    def test_dispatch_healpix(self):
        alm = _random_alm()
        m = harmonic.alm2map(alm, spin=0, nside=NSIDE, lmax=LMAX)
        assert m.shape[-1] == hp.nside2npix(NSIDE)
        assert not isinstance(m, enmap.ndmap)

    def test_dispatch_car_via_shape_wcs(self, car_geometry):
        shape, wcs = car_geometry
        alm = _random_alm()
        m = harmonic.alm2map(alm, spin=0, shape=shape, wcs=wcs, lmax=LMAX)
        assert isinstance(m, enmap.ndmap)
        assert m.shape[-2:] == tuple(shape[-2:])

    def test_dispatch_car_via_out(self, car_geometry):
        shape, wcs = car_geometry
        out = enmap.zeros(shape, wcs=wcs)
        alm = _random_alm()
        m = harmonic.alm2map(alm, spin=0, out=out, lmax=LMAX)
        assert isinstance(m, enmap.ndmap)
        assert m is out
        assert not np.all(out == 0)

    def test_healpix_spin0_scalar(self):
        alm = _random_alm()
        m = harmonic.alm2map(alm, spin=0, nside=NSIDE, lmax=LMAX)
        assert m.shape == (hp.nside2npix(NSIDE),)

    def test_healpix_spin0_batch(self):
        ncomp = 4
        alms = _random_alm(ncomp=ncomp)
        m = harmonic.alm2map(alms, spin=0, nside=NSIDE, lmax=LMAX)
        assert m.shape == (ncomp, hp.nside2npix(NSIDE))
        for i in range(ncomp):
            assert_allclose(m[i], harmonic.alm2map(alms[i], spin=0, nside=NSIDE, lmax=LMAX))

    def test_healpix_list_spin(self):
        alm_T = _random_alm()
        alm_QU = _random_alm(ncomp=2)
        alms_teb = np.concatenate([alm_T[None], alm_QU], axis=0)
        m = harmonic.alm2map(alms_teb, spin=[0, 2], nside=NSIDE, lmax=LMAX)
        assert m.shape == (3, hp.nside2npix(NSIDE))
        assert_allclose(m[0], harmonic.alm2map(alm_T, spin=0, nside=NSIDE, lmax=LMAX))
        assert_allclose(m[1:], harmonic.alm2map(alm_QU, spin=2, nside=NSIDE, lmax=LMAX))

    def test_car_list_spin(self, car_geometry):
        shape, wcs = car_geometry
        alm_T = _random_alm()
        alm_QU = _random_alm(ncomp=2)
        alms_teb = np.concatenate([alm_T[None], alm_QU], axis=0)
        m = harmonic.alm2map(alms_teb, spin=[0, 2], shape=shape[-2:], wcs=wcs, lmax=LMAX)
        assert isinstance(m, enmap.ndmap)
        assert m.shape == (3, *shape[-2:])
        assert_allclose(m[0], harmonic.alm2map(alm_T, spin=0, shape=shape[-2:], wcs=wcs, lmax=LMAX))
        assert_allclose(
            m[1:], harmonic.alm2map(alm_QU, spin=2, shape=shape[-2:], wcs=wcs, lmax=LMAX)
        )

    def test_healpix_out_spin0(self):
        alm = _random_alm()
        out = np.zeros(hp.nside2npix(NSIDE))
        m = harmonic.alm2map(alm, spin=0, nside=NSIDE, out=out, lmax=LMAX)
        assert m is out
        assert not np.all(out == 0)
        assert_allclose(m, harmonic.alm2map(alm, spin=0, nside=NSIDE, lmax=LMAX))

    def test_healpix_out_spin2(self):
        alm = _random_alm(ncomp=2)
        out = np.zeros((2, hp.nside2npix(NSIDE)))
        m = harmonic.alm2map(alm, spin=2, nside=NSIDE, out=out, lmax=LMAX)
        assert m is out
        assert not np.all(out == 0)
        assert_allclose(m, harmonic.alm2map(alm, spin=2, nside=NSIDE, lmax=LMAX))

    def test_healpix_out_list_spin(self):
        alm_T = _random_alm()
        alm_QU = _random_alm(ncomp=2)
        alms = np.concatenate([alm_T[None], alm_QU], axis=0)
        out = np.zeros((3, hp.nside2npix(NSIDE)))
        m = harmonic.alm2map(alms, spin=[0, 2], nside=NSIDE, out=out, lmax=LMAX)
        assert m is out
        assert not np.all(out == 0)
        assert_allclose(m, harmonic.alm2map(alms, spin=[0, 2], nside=NSIDE, lmax=LMAX))

    def test_requires_target(self):
        alm = _random_alm()
        with pytest.raises(ValueError):
            harmonic.alm2map(alm, spin=0)

    def test_rejects_both_targets(self, car_geometry):
        shape, wcs = car_geometry
        alm = _random_alm()
        with pytest.raises(ValueError):
            harmonic.alm2map(alm, spin=0, nside=NSIDE, shape=shape, wcs=wcs)

    def test_rejects_out_and_shape_wcs_car(self, car_geometry):
        shape, wcs = car_geometry
        alm = _random_alm()
        out = enmap.zeros(shape, wcs=wcs)
        with pytest.raises(ValueError, match="out"):
            harmonic.alm2map(alm, spin=0, out=out, shape=shape, wcs=wcs)

    def test_rejects_out_and_shape_wcs_healpix(self, car_geometry):
        shape, wcs = car_geometry
        alm = _random_alm()
        out = np.zeros(hp.nside2npix(NSIDE))
        with pytest.raises(ValueError, match="out"):
            harmonic.alm2map(alm, spin=0, out=out, shape=shape, wcs=wcs)

    def test_lmax_exceeds_alm_bandlimit(self):
        alm = _random_alm()
        with pytest.raises(ValueError, match="lmax"):
            harmonic.alm2map(alm, spin=0, nside=NSIDE, lmax=LMAX + 5)

    def test_healpix_out_batched_spin0(self):
        ncomp = 3
        alms = _random_alm(ncomp=ncomp)
        out = np.zeros((ncomp, hp.nside2npix(NSIDE)))
        m = harmonic.alm2map(alms, spin=0, nside=NSIDE, out=out, lmax=LMAX)
        assert m is out
        assert_allclose(m, harmonic.alm2map(alms, spin=0, nside=NSIDE, lmax=LMAX))

    def test_healpix_nthreads(self):
        alm = _random_alm()
        m1 = harmonic.alm2map(alm, spin=0, nside=NSIDE, lmax=LMAX, nthreads=1)
        m2 = harmonic.alm2map(alm, spin=0, nside=NSIDE, lmax=LMAX, nthreads=2)
        assert_allclose(m1, m2, rtol=1e-10)


# ---------------------------------------------------------------------------
# almxfl
# ---------------------------------------------------------------------------


class TestAlmxfl:
    def test_1d_passthrough(self):
        alm = _random_alm()
        fl = np.linspace(1.0, 2.0, LMAX + 1)
        assert_allclose(harmonic.almxfl(alm, fl), hp.almxfl(alm.copy(), fl))

    def test_2d_per_row(self):
        alms = _random_alm(ncomp=3)
        fl = np.linspace(1.0, 2.0, LMAX + 1)
        out = harmonic.almxfl(alms, fl)
        assert out.shape == alms.shape
        for i in range(3):
            assert_allclose(out[i], hp.almxfl(alms[i].copy(), fl))

    def test_inplace(self):
        alms = _random_alm(ncomp=2)
        snapshot = alms.copy()
        fl = np.linspace(1.0, 2.0, LMAX + 1)
        out = harmonic.almxfl(alms, fl, inplace=True)
        assert out is alms
        for i in range(2):
            assert_allclose(alms[i], hp.almxfl(snapshot[i].copy(), fl))

    def test_no_inplace_does_not_mutate(self):
        alms = _random_alm(ncomp=2)
        snapshot = alms.copy()
        fl = np.linspace(1.0, 2.0, LMAX + 1)
        harmonic.almxfl(alms, fl, inplace=False)
        assert_allclose(alms, snapshot)


# ---------------------------------------------------------------------------
# anafast
# ---------------------------------------------------------------------------


class TestAnafast:
    def test_healpix_scalar(self):
        m = RNG.standard_normal(hp.nside2npix(NSIDE))
        cl_us = harmonic.anafast(m, lmax=LMAX, pol=False, niter=3)
        cl_hp = hp.anafast(m, lmax=LMAX, pol=False, iter=3)
        assert_allclose(cl_us, cl_hp, rtol=1e-10)

    def test_healpix_pol(self):
        tqu = RNG.standard_normal((3, hp.nside2npix(NSIDE)))
        cl_us = harmonic.anafast(tqu, lmax=LMAX, pol=True, niter=3)
        cl_hp = hp.anafast(tqu, lmax=LMAX, pol=True, iter=3)
        assert cl_us.shape == cl_hp.shape  # (6, lmax+1)
        assert_allclose(cl_us, cl_hp, rtol=1e-10)

    def test_healpix_cross(self):
        npix = hp.nside2npix(NSIDE)
        m1 = RNG.standard_normal(npix)
        m2 = RNG.standard_normal(npix)
        cl_us = harmonic.anafast(m1, m2, lmax=LMAX, pol=False, niter=3)
        cl_hp = hp.anafast(m1, map2=m2, lmax=LMAX, pol=False, iter=3)
        assert_allclose(cl_us, cl_hp, rtol=1e-10)

    def test_car_auto(self, car_geometry):
        shape, wcs = car_geometry
        m = enmap.enmap(RNG.standard_normal(shape), wcs)
        cl = harmonic.anafast(m, lmax=LMAX)
        assert cl.shape[-1] == LMAX + 1

    def test_car_cross(self, car_geometry):
        shape, wcs = car_geometry
        m1 = enmap.enmap(RNG.standard_normal(shape), wcs)
        m2 = enmap.enmap(RNG.standard_normal(shape), wcs)
        cl = harmonic.anafast(m1, m2, lmax=LMAX)
        assert cl.shape[-1] == LMAX + 1

    def test_pol_wrong_nstokes_healpix(self):
        m = RNG.standard_normal((2, hp.nside2npix(NSIDE)))
        with pytest.raises(ValueError, match="pol=True"):
            harmonic.anafast(m, lmax=LMAX, pol=True)

    def test_pol_wrong_nstokes_car(self, car_geometry):
        shape, wcs = car_geometry
        m = enmap.enmap(RNG.standard_normal((2, *shape[-2:])), wcs)
        with pytest.raises(ValueError, match="pol=True"):
            harmonic.anafast(m, lmax=LMAX, pol=True)

    def test_healpix_pol_cross(self):
        tqu1 = RNG.standard_normal((3, hp.nside2npix(NSIDE)))
        tqu2 = RNG.standard_normal((3, hp.nside2npix(NSIDE)))
        cl_us = harmonic.anafast(tqu1, tqu2, lmax=LMAX, pol=True, niter=3)
        cl_hp = hp.anafast(tqu1, map2=tqu2, lmax=LMAX, pol=True, iter=3)
        assert cl_us.shape == cl_hp.shape
        assert_allclose(cl_us, cl_hp, rtol=1e-10)


# ---------------------------------------------------------------------------
# synfast
# ---------------------------------------------------------------------------

NSIDE_SF = 64
LMAX_SF = 2 * NSIDE_SF
NPIX_SF = hp.nside2npix(NSIDE_SF)
LMAX_CAR = 20


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


def _cls6_to_cov(cls6):
    """Build (3, 3, nl) covariance matrix from a (6, nl) flat diagonal array."""
    cov = np.zeros((3, 3, cls6.shape[1]))
    cov[0, 0] = cls6[0]
    cov[1, 1] = cls6[1]
    cov[2, 2] = cls6[2]
    cov[0, 1] = cov[1, 0] = cls6[3]
    cov[1, 2] = cov[2, 1] = cls6[4]
    cov[0, 2] = cov[2, 0] = cls6[5]
    return cov


_CL_T = _make_cls(LMAX_SF)[0]
_CL_POL = _make_cls(LMAX_SF)
_CL_6_CAR = _make_cls(LMAX_CAR)
_CL_TT_CAR = _CL_6_CAR[0]


class TestSynfast:
    def test_rejects_both_targets(self, car_geometry):
        shape, wcs = car_geometry
        with pytest.raises(ValueError):
            harmonic.synfast(np.ones(LMAX + 1), nside=NSIDE, shape=shape, wcs=wcs)

    def test_requires_target(self):
        with pytest.raises(ValueError):
            harmonic.synfast(np.ones(LMAX + 1))

    def test_healpix_t_shape(self):
        m = harmonic.synfast(_CL_T, nside=NSIDE_SF, lmax=LMAX_SF, seed=1)
        assert m.shape == (NPIX_SF,)
        assert m.dtype.kind == "f"
        assert np.all(np.isfinite(m))

    def test_healpix_pol_shape(self):
        m = harmonic.synfast(_CL_POL, nside=NSIDE_SF, lmax=LMAX_SF, seed=2)
        assert m.shape == (3, NPIX_SF)
        assert m.dtype.kind == "f"
        assert np.all(np.isfinite(m))

    def test_healpix_t_seed_reproducible(self):
        m1 = harmonic.synfast(_CL_T, nside=NSIDE_SF, lmax=LMAX_SF, seed=42)
        m2 = harmonic.synfast(_CL_T, nside=NSIDE_SF, lmax=LMAX_SF, seed=42)
        assert np.array_equal(m1, m2)

    def test_healpix_pol_seed_reproducible(self):
        m1 = harmonic.synfast(_CL_POL, nside=NSIDE_SF, lmax=LMAX_SF, seed=42)
        m2 = harmonic.synfast(_CL_POL, nside=NSIDE_SF, lmax=LMAX_SF, seed=42)
        assert np.array_equal(m1, m2)

    def test_healpix_t_matches_hp_synfast(self):
        np.random.seed(99)  # noqa: NPY002
        expected = hp.synfast(_CL_T, nside=NSIDE_SF, lmax=LMAX_SF, new=True)
        result = harmonic.synfast(_CL_T, nside=NSIDE_SF, lmax=LMAX_SF, seed=99)
        assert_allclose(result, expected, rtol=1e-10)

    def test_healpix_pol_matches_hp_synfast(self):
        np.random.seed(99)  # noqa: NPY002
        expected = hp.synfast(_CL_POL, nside=NSIDE_SF, lmax=LMAX_SF, new=True)
        result = harmonic.synfast(_CL_POL, nside=NSIDE_SF, lmax=LMAX_SF, seed=99)
        assert_allclose(result, expected, rtol=1e-10)

    def test_healpix_cov_matrix_shape(self):
        cov = _cls6_to_cov(_CL_6_CAR)
        m = harmonic.synfast(cov, nside=NSIDE_SF, lmax=LMAX_SF, seed=5)
        assert m.shape == (3, NPIX_SF)
        assert np.all(np.isfinite(m))

    def test_healpix_cov_matches_flat(self):
        cov = _cls6_to_cov(_CL_POL)
        m_flat = harmonic.synfast(_CL_POL, nside=NSIDE_SF, lmax=LMAX_SF, seed=6)
        m_cov = harmonic.synfast(cov, nside=NSIDE_SF, lmax=LMAX_SF, seed=6)
        assert_array_equal(m_flat, m_cov)

    def test_car_1d_shape(self, car_geometry):
        shape, wcs = car_geometry
        m = harmonic.synfast(_CL_TT_CAR, shape=shape, wcs=wcs, lmax=LMAX_CAR, seed=1)
        assert isinstance(m, enmap.ndmap)
        assert np.all(np.isfinite(m))

    def test_car_flat_6spec_shape(self, car_tqu_geometry):
        tqu_shape, wcs = car_tqu_geometry
        m = harmonic.synfast(_CL_6_CAR, shape=tqu_shape, wcs=wcs, lmax=LMAX_CAR, seed=2)
        assert isinstance(m, enmap.ndmap)
        assert m.shape[-3] == 3
        assert np.all(np.isfinite(m))

    def test_car_4spec_shape(self, car_tqu_geometry):
        tqu_shape, wcs = car_tqu_geometry
        m = harmonic.synfast(_CL_6_CAR[:4], shape=tqu_shape, wcs=wcs, lmax=LMAX_CAR, seed=3)
        assert isinstance(m, enmap.ndmap)
        assert m.shape[-3] == 3
        assert np.all(np.isfinite(m))

    def test_car_cov_matrix_shape(self, car_tqu_geometry):
        tqu_shape, wcs = car_tqu_geometry
        cov = _cls6_to_cov(_CL_6_CAR)
        m = harmonic.synfast(cov, shape=tqu_shape, wcs=wcs, lmax=LMAX_CAR, seed=4)
        assert isinstance(m, enmap.ndmap)
        assert m.shape[-3] == 3
        assert np.all(np.isfinite(m))

    def test_car_flat_and_cov_equivalent(self, car_tqu_geometry):
        tqu_shape, wcs = car_tqu_geometry
        cov = _cls6_to_cov(_CL_6_CAR)
        m_flat = harmonic.synfast(_CL_6_CAR, shape=tqu_shape, wcs=wcs, lmax=LMAX_CAR, seed=7)
        m_cov = harmonic.synfast(cov, shape=tqu_shape, wcs=wcs, lmax=LMAX_CAR, seed=7)
        assert_array_equal(m_flat, m_cov)

    def test_car_invalid_flat_raises(self, car_geometry):
        shape, wcs = car_geometry
        with pytest.raises(ValueError, match="triangular"):
            harmonic.synfast(np.ones((5, LMAX_CAR + 1)), shape=shape, wcs=wcs, lmax=LMAX_CAR)

    def test_healpix_nthreads(self):
        m1 = harmonic.synfast(_CL_T, nside=NSIDE_SF, lmax=LMAX_SF, seed=10, nthreads=1)
        m2 = harmonic.synfast(_CL_T, nside=NSIDE_SF, lmax=LMAX_SF, seed=10, nthreads=2)
        assert_allclose(m1, m2, rtol=1e-10)


# ---------------------------------------------------------------------------
# _normalise_cl
# ---------------------------------------------------------------------------

NL = LMAX + 1


class TestNormaliseCl:
    def test_1d_passthrough(self):
        cl = np.ones(NL)
        out = harmonic._normalise_cl(cl)
        assert isinstance(out, np.ndarray) and out.ndim == 1
        assert_array_equal(out, cl)

    def test_2d_returns_list(self):
        cl = np.arange(6 * NL, dtype=float).reshape(6, NL)
        out = harmonic._normalise_cl(cl)
        assert isinstance(out, list) and len(out) == 6
        for i in range(6):
            assert_array_equal(out[i], cl[i])

    def test_4spec_passthrough(self):
        cl = np.arange(4 * NL, dtype=float).reshape(4, NL)
        out = harmonic._normalise_cl(cl)
        assert isinstance(out, list) and len(out) == 4
        for i in range(4):
            assert_array_equal(out[i], cl[i])

    def test_3d_diagonal_order(self):
        cov = np.zeros((3, 3, NL))
        for i in range(3):
            for j in range(3):
                cov[i, j] = (i * 3 + j + 1) * np.ones(NL)
        out = harmonic._normalise_cl(cov)
        # Diagonal order: TT EE BB TE EB TB
        assert_array_equal(out[0], cov[0, 0])  # TT
        assert_array_equal(out[1], cov[1, 1])  # EE
        assert_array_equal(out[2], cov[2, 2])  # BB
        assert_array_equal(out[3], cov[0, 1])  # TE
        assert_array_equal(out[4], cov[1, 2])  # EB
        assert_array_equal(out[5], cov[0, 2])  # TB

    def test_invalid_nspec_raises(self):
        with pytest.raises(ValueError, match="triangular"):
            harmonic._normalise_cl(np.ones((5, NL)))


# ---------------------------------------------------------------------------
# getlmax
# ---------------------------------------------------------------------------


class TestGetlmax:
    def test_1d(self):
        assert harmonic.getlmax(np.zeros(NALM, dtype=complex)) == LMAX

    def test_2d_uses_last_axis(self):
        assert harmonic.getlmax(np.zeros((3, NALM), dtype=complex)) == LMAX
