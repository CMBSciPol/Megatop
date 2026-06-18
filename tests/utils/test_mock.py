import healpy as hp
import numpy as np
import pytest
from pixell import enmap, utils

from megatop.config import (
    CARConfig,
    Config,
    DataDirsConfig,
    FiducialCMBConfig,
    GeneralConfig,
    MapSetConfig,
    OutputDirsConfig,
    PixelisationConfig,
)
from megatop.landscapes import HealpixLandscape
from megatop.utils import mock

NSIDE = 8
RNG = np.random.default_rng()
HEALPIX = HealpixLandscape(NSIDE)


def make_car_config(tmp_path, lmax=2 * 16):
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
    return cfg, shape


class _FakeEmission:
    def __init__(self, arr):
        self.value = arr


class _FakeSky:
    """Offline stand-in for `pysm3.Sky` that records how often it is built."""

    call_count = 0
    last_nside = None

    def __init__(self, nside, preset_strings=None, output_unit=None):
        type(self).call_count += 1
        type(self).last_nside = nside
        self.nside = nside

    def get_emission(self, freq, weights=None):
        return _FakeEmission(RNG.random((3, hp.nside2npix(self.nside))))


@pytest.fixture
def fake_pysm_sky(monkeypatch):
    """Replace `pysm3.Sky` with an offline stub.

    The real `Sky` downloads template FITS files from a flaky remote (NERSC
    portal), which makes the foreground tests slow and network-bound. The stub
    returns random HEALPix emission of the right shape so the tests still
    exercise megatop's wrapper (`reproject`/`stack`/landscape handling) without
    any network access. Tests assert `_FakeSky.call_count` to prove the patch
    actually took effect (otherwise a passing test could be silently hitting
    the network or a local astropy cache).
    """
    _FakeSky.call_count = 0
    monkeypatch.setattr(mock, "Sky", _FakeSky)


def test_fixed_cmb_simulation():
    cl_cmb_model = RNG.random((6, 3 * NSIDE - 1))
    maps_1 = mock.generate_map_cmb(cl_cmb_model, HEALPIX, lmax=2 * NSIDE, cmb_seed=1234)
    maps_2 = mock.generate_map_cmb(cl_cmb_model, HEALPIX, lmax=2 * NSIDE, cmb_seed=1234)
    assert np.all(maps_1 == maps_2)


def test_shape_white_noise_map():
    freq_maps = mock.get_noise_map_from_white_noise(1.0, HEALPIX)
    assert freq_maps.shape == (3, hp.nside2npix(NSIDE))


def test_white_noise_reproducible():
    a = mock.get_noise_map_from_white_noise(2.0, HEALPIX, seed=[1, 2])
    b = mock.get_noise_map_from_white_noise(2.0, HEALPIX, seed=[1, 2])
    assert np.array_equal(a, b)


def test_shape_spectra_noise_map():
    lmax = 2 * NSIDE
    nell = np.arange(2, lmax + 1)
    freq_maps = mock.get_noise_map_from_noise_spectra(nell, lmax, HEALPIX)
    assert freq_maps.shape == (3, hp.nside2npix(NSIDE))


@pytest.mark.parametrize(
    ("sky_model", "working", "expected"),
    [
        (["d0", "s0"], 256, 512),  # PySM2 nside-512 templates -> floor 512
        (["d0", "s0"], 1024, 1024),  # working above floor -> render at working
        (["d8", "s3"], 256, 512),
        (["d10"], 256, 2048),  # PySM3 template model: smallest available_nside 2048
        (["d0", "s5"], 256, 2048),  # any 2048-floor component promotes the render nside
        (["d11"], 256, 256),  # realization model: synthesized at the working nside, no floor
        (["d11", "s6"], 512, 512),  # realizations only -> render at working
    ],
)
def test_pysm_render_nside(sky_model, working, expected):
    assert mock.pysm_render_nside(sky_model, working) == expected


@pytest.mark.usefixtures("fake_pysm_sky")
def test_shape_fg_map():
    map_sets = [
        MapSetConfig(freq_tag=100, exp_tag="test", nhits_map_path="SO_nominal", beam=10.0),
        MapSetConfig(freq_tag=200, exp_tag="test", nhits_map_path="SO_nominal", beam=10.0),
    ]
    map_sets[0].frequency = 100
    map_sets[0].weight = 1
    map_sets[1].frequency = 200
    map_sets[1].weight = 1
    freq_maps = mock.generate_map_fgs_pysm(map_sets, NSIDE, 2 * NSIDE, ["d0"], HEALPIX)
    assert _FakeSky.call_count == 1  # the stub ran, not the real pysm3.Sky
    assert (
        _FakeSky.last_nside == 512
    )  # rendered at the d0 template native (512), not the output nside
    assert freq_maps.shape == (2, 3, hp.nside2npix(NSIDE))


def test_hit_map():
    noise_maps = np.ones((2, 3, hp.nside2npix(NSIDE)))
    binary = np.ones(hp.nside2npix(NSIDE))
    nhits_map = RNG.random(hp.nside2npix(NSIDE))
    noise_maps_rescaled = mock.include_hits_noise(
        noise_maps=noise_maps, common_nhits_map=nhits_map, binary_mask=binary
    )
    assert np.all(noise_maps_rescaled >= np.ones((2, 3, hp.nside2npix(NSIDE))))
    assert np.all(np.isfinite(noise_maps_rescaled))
    assert noise_maps_rescaled.shape == (2, 3, hp.nside2npix(NSIDE))
    nhits_map[RNG.integers(0, hp.nside2npix(NSIDE))] = 0.0
    with pytest.raises(FloatingPointError):
        _ = mock.include_hits_noise(
            noise_maps=noise_maps, common_nhits_map=nhits_map, binary_mask=binary
        )


def test_beam():
    freq_map = RNG.random((3, hp.nside2npix(NSIDE)))
    freq_map_beamed = mock.beam_winpix_correction(
        NSIDE, freq_map=freq_map, beam_FWHM=100, lmax=2 * NSIDE
    )
    assert np.all(np.isfinite(freq_map_beamed))
    assert freq_map.shape == freq_map_beamed.shape


def test_apply_obsmat():
    freq_map = RNG.random((3, hp.nside2npix(NSIDE)))

    def obsmat_fun(x):
        return x

    freq_map_fun = mock.apply_observation_matrix(obsmat_fun, freq_map)
    assert np.all(freq_map_fun == freq_map)


# --- CAR mock paths ----------------------------------------------------------


def test_cmb_car(tmp_path):
    cfg, shape = make_car_config(tmp_path)
    cl = RNG.random((6, 3 * cfg.nside - 1))
    m = mock.generate_map_cmb(cl, cfg.landscape, lmax=cfg.lmax, cmb_seed=1234)
    assert isinstance(m, enmap.ndmap)
    assert m.shape == (3, *shape[-2:])
    assert np.all(np.isfinite(m))


def test_white_noise_car(tmp_path):
    cfg, shape = make_car_config(tmp_path)
    m = mock.get_noise_map_from_white_noise(1.0, cfg.landscape, seed=0)
    assert isinstance(m, enmap.ndmap)
    assert m.shape == (3, *shape[-2:])
    assert np.all(np.isfinite(m))


def test_spectra_noise_car(tmp_path):
    cfg, shape = make_car_config(tmp_path)
    nell = np.arange(2, cfg.lmax + 1)
    m = mock.get_noise_map_from_noise_spectra(nell, cfg.lmax, cfg.landscape, seed=0)
    assert isinstance(m, enmap.ndmap)
    assert m.shape == (3, *shape[-2:])


def test_beam_car(tmp_path):
    cfg, shape = make_car_config(tmp_path)
    freq_map = enmap.enmap(RNG.random((3, *shape[-2:])), cfg.geometry[1])
    beamed = mock.beam_winpix_correction(cfg.nside, freq_map, beam_FWHM=30, lmax=cfg.lmax)
    assert isinstance(beamed, enmap.ndmap)
    assert beamed.shape == freq_map.shape
    assert np.all(np.isfinite(beamed))


def test_fg_car(tmp_path):
    cfg, shape = make_car_config(tmp_path)
    map_sets = [MapSetConfig(freq_tag=100, exp_tag="test", nhits_map_path="SO_nominal", beam=10.0)]
    map_sets[0].frequency = 100
    map_sets[0].weight = 1
    fg = mock.generate_map_fgs_pysm(map_sets, cfg.nside, cfg.lmax, ["d0"], cfg.landscape)
    assert isinstance(fg, enmap.ndmap)
    assert fg.shape == (1, 3, *shape[-2:])
