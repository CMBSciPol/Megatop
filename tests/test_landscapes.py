import healpy as hp
import numpy as np
import pytest
from pixell import enmap, utils

import megatop.landscapes as land
from megatop.config import (
    CARConfig,
    Config,
    DataDirsConfig,
    FiducialCMBConfig,
    GeneralConfig,
    HealpixConfig,
    OutputDirsConfig,
    PixelisationConfig,
)

NSIDE = 32


def make_car_config(tmp_path, lmax=2 * NSIDE):
    """Minimal CAR Config backed by a small equatorial patch geometry file."""
    shape, wcs = enmap.geometry(
        pos=[[-5 * utils.degree, -5 * utils.degree], [5 * utils.degree, 5 * utils.degree]],
        res=0.5 * utils.degree,
        proj="car",
    )
    geom_file = tmp_path / "geom.fits"
    enmap.write_map_geometry(str(geom_file), shape, wcs)
    cfg = Config(
        data_dirs=DataDirsConfig(root=str(tmp_path / "data")),
        output_dirs=OutputDirsConfig(root=str(tmp_path / "out")),
        fiducial_cmb=FiducialCMBConfig(),
        general_pars=GeneralConfig(
            pixelisation=PixelisationConfig(car=CARConfig(geometry_file=str(geom_file))),
            lmax=lmax,
        ),
        map_sets=[],
    )
    return cfg, shape, wcs


def make_healpix_config(nside=NSIDE, lmax=2 * NSIDE):
    return Config(
        data_dirs=DataDirsConfig(root="data"),
        output_dirs=OutputDirsConfig(root="out"),
        fiducial_cmb=FiducialCMBConfig(),
        general_pars=GeneralConfig(
            pixelisation=PixelisationConfig(healpix=HealpixConfig(nside=nside)), lmax=lmax
        ),
        map_sets=[],
    )


# --- config geometry / landscape dispatch ---------------------------------


def test_car_config_requires_geometry_file():
    # geometry_file is a required field on CARConfig
    with pytest.raises(ValueError, match="geometry_file"):
        CARConfig()


def test_landscape_requires_exactly_one():
    with pytest.raises(ValueError, match="exactly one"):
        PixelisationConfig()
    with pytest.raises(ValueError, match="exactly one"):
        PixelisationConfig(healpix=HealpixConfig(), car=CARConfig(geometry_file="x.fits"))


def test_config_geometry_returns_shape_wcs(tmp_path):
    cfg, shape, wcs = make_car_config(tmp_path)
    g_shape, g_wcs = cfg.geometry
    assert tuple(g_shape) == tuple(shape)
    assert cfg.is_car


def test_healpix_config_geometry_raises():
    cfg = make_healpix_config()
    assert not cfg.is_car
    with pytest.raises(ValueError, match="CAR"):
        _ = cfg.geometry


def test_config_landscape_dispatch(tmp_path):
    cfg_hp = make_healpix_config()
    assert isinstance(cfg_hp.landscape, land.HealpixLandscape)
    assert cfg_hp.landscape.nside == NSIDE

    cfg_car, shape, wcs = make_car_config(tmp_path)
    p = cfg_car.landscape
    assert isinstance(p, land.CARLandscape)
    assert tuple(p.shape) == tuple(shape)


def test_config_landscape_is_cached(tmp_path):
    # cached_property on a pydantic model must return the same instance each time
    cfg = make_healpix_config()
    assert cfg.landscape is cfg.landscape

    cfg_car, _, _ = make_car_config(tmp_path)
    assert cfg_car.landscape is cfg_car.landscape


# --- pixel_area / zeros ------------------------------------------------------


def test_pixel_area_healpix_scalar():
    area = land.HealpixLandscape(NSIDE).pixel_area_arcmin2()
    assert np.isscalar(area) or np.ndim(area) == 0
    assert area == pytest.approx(hp.nside2resol(NSIDE, arcmin=True) ** 2)


def test_pixel_area_car_map(tmp_path):
    _, shape, wcs = make_car_config(tmp_path)
    area = land.CARLandscape(shape, wcs).pixel_area_arcmin2()
    assert area.shape == shape[-2:]
    assert np.all(area > 0)


def test_zeros_shapes(tmp_path):
    z = land.HealpixLandscape(NSIDE).zeros((3,))
    assert z.shape == (3, hp.nside2npix(NSIDE))

    _, shape, wcs = make_car_config(tmp_path)
    zc = land.CARLandscape(shape, wcs).zeros((2, 3))
    assert isinstance(zc, enmap.ndmap)
    assert zc.shape == (2, 3, *shape[-2:])


# --- project (reproject onto target geometry) --------------------------------


def test_project_car_harmonic_preserves_monopole(tmp_path):
    _, shape, wcs = make_car_config(tmp_path)
    m = np.zeros((3, hp.nside2npix(NSIDE)))
    m[0] = 1.0  # constant T (monopole only); Q=U=0
    out = land.CARLandscape(shape, wcs).reproject_harmonic(m, spin=(0, 2), rot=None)
    assert isinstance(out, enmap.ndmap)
    assert out.shape == (3, *shape[-2:])
    assert out[0].mean() == pytest.approx(1.0, abs=1e-3)


def test_project_car_pixel_keeps_mask_bounded(tmp_path):
    _, shape, wcs = make_car_config(tmp_path)
    mask = np.ones(hp.nside2npix(NSIDE))
    out = land.CARLandscape(shape, wcs).reproject_pixel(mask, rot=None)
    assert np.all(out >= -1e-6)
    assert np.all(out <= 1 + 1e-6)


def test_project_healpix_pixel_resamples():
    # no rotation: a HEALPix reproject is a resample to the target nside
    src = np.arange(hp.nside2npix(NSIDE), dtype=np.float64)
    out = land.HealpixLandscape(NSIDE // 2).reproject_pixel(src, rot=None)
    assert out.shape == (hp.nside2npix(NSIDE // 2),)


def test_project_healpix_extensive_conserves_sum():
    # extensive=True must conserve the total (hit counts), unlike the default mean
    src = np.arange(hp.nside2npix(NSIDE), dtype=np.float64)
    p = land.HealpixLandscape(NSIDE // 2)
    intensive = p.reproject_pixel(src, rot=None)
    extensive = p.reproject_pixel(src, rot=None, extensive=True)
    assert extensive.sum() == pytest.approx(src.sum())
    assert not np.allclose(intensive, extensive)


# --- read/write round trip ---------------------------------------------------


def test_write_read_roundtrip_car(tmp_path):
    _, shape, wcs = make_car_config(tmp_path)
    p = land.CARLandscape(shape, wcs)
    m = enmap.enmap(
        np.arange(3 * shape[-2] * shape[-1], dtype=np.float64).reshape((3, *shape[-2:])), wcs
    )
    path = tmp_path / "m.fits"
    p.write_map(path, m)
    back = p.read_map(path)
    assert isinstance(back, enmap.ndmap)
    assert back.dtype == m.dtype
    assert np.array_equal(back, m)


def test_write_read_roundtrip_healpix(tmp_path):
    p = land.HealpixLandscape(NSIDE)
    m = np.random.default_rng(0).random(hp.nside2npix(NSIDE))
    path = tmp_path / "m.fits"
    p.write_map(path, m, dtype=np.float64)
    back = p.read_map(path)
    assert back.dtype == m.dtype
    assert np.array_equal(back, m)


# --- stack -------------------------------------------------------------------


def test_stack_healpix():
    maps = [np.full(hp.nside2npix(NSIDE), i, dtype=np.float64) for i in range(2)]
    stacked = land.HealpixLandscape(NSIDE).stack(maps)
    assert stacked.shape == (2, hp.nside2npix(NSIDE))


def test_stack_car_preserves_wcs(tmp_path):
    _, shape, wcs = make_car_config(tmp_path)
    p = land.CARLandscape(shape, wcs)
    maps = [p.zeros((3,)) + i for i in range(2)]
    stacked = p.stack(maps)
    assert isinstance(stacked, enmap.ndmap)
    assert stacked.shape == (2, 3, *shape[-2:])
    assert stacked.wcs is not None


# --- working nside ---


@pytest.mark.parametrize(
    ("lmax", "expected"),
    [
        (1, 1),
        (2, 1),
        (3, 2),
        (4, 2),
        (5, 4),
        (250, 128),
        (500, 256),
        (512, 256),
        (513, 512),
        (1024, 512),
    ],
)
def test_car_working_nside(lmax, expected):
    """CAR working nside: smallest power-of-two with lmax <= 2 * nside (pysm render res)."""
    shape, wcs = enmap.geometry(
        pos=[[-1 * utils.degree, -1 * utils.degree], [1 * utils.degree, 1 * utils.degree]],
        res=0.5 * utils.degree,
        proj="car",
    )
    nside = land.CARLandscape(shape, wcs).working_nside(lmax)
    assert nside == expected
    # invariant: bound holds and it is the smallest such power of two
    assert lmax <= 2 * nside
    assert nside == 1 or lmax > 2 * (nside // 2)
    assert nside & (nside - 1) == 0  # power of two
