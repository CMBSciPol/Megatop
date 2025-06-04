import healpy as hp
import numpy as np
import pytest

from megatop.config import MapSetConfig
from megatop.utils import mock

NSIDE = 8
RNG = np.random.default_rng()


def test_fixed_cmb_simulation():
    cl_cmb_model = RNG.random((1, 6, 3 * NSIDE - 1))
    maps_1 = mock.generate_map_cmb(cl_cmb_model, NSIDE, cmb_seed=1234)
    maps_2 = mock.generate_map_cmb(cl_cmb_model, NSIDE, cmb_seed=1234)
    assert np.all(maps_1 == maps_2)


def test_shape_white_noise_map():
    freq_maps = mock.get_noise_map_from_white_noise([100, 200], NSIDE, [2, 3])
    assert freq_maps.shape == (2, 3, hp.nside2npix(NSIDE))


def test_shape_spectra_noise_map():
    nell = np.arange(2, 3 * NSIDE - 1)
    freq_maps = mock.get_noise_map_from_noise_spectra([100, 200], NSIDE, nell)
    assert freq_maps.shape == (2, 3, hp.nside2npix(NSIDE))


def test_shape_fg_map():
    map_sets = [
        MapSetConfig(freq_tag=100, exp_tag="test"),
        MapSetConfig(freq_tag=200, exp_tag="test"),
    ]
    map_sets[0].frequency = 100
    map_sets[0].weight = 1
    map_sets[1].frequency = 200
    map_sets[1].weight = 1
    freq_maps = mock.generate_map_fgs_pysm(
        map_sets, NSIDE, ["d0"], input_coord="G", output_coord="E"
    )
    assert freq_maps.shape == (2, 3, hp.nside2npix(NSIDE))


def test_hit_map():
    noise_maps = np.ones((2, 3, hp.nside2npix(NSIDE)))
    binary = np.ones(hp.nside2npix(NSIDE))
    nhits_map = RNG.random(hp.nside2npix(NSIDE))
    noise_maps_rescaled = mock.include_hits_noise(
        noise_maps=noise_maps, nhits_map=nhits_map, binary_mask=binary
    )
    assert np.all(noise_maps_rescaled >= np.ones((2, 3, hp.nside2npix(NSIDE))))
    assert np.all(np.isfinite(noise_maps_rescaled))
    assert noise_maps_rescaled.shape == (2, 3, hp.nside2npix(NSIDE))
    nhits_map[RNG.integers(0, hp.nside2npix(NSIDE))] = 0.0
    with pytest.raises(FloatingPointError):
        _ = mock.include_hits_noise(noise_maps=noise_maps, nhits_map=nhits_map, binary_mask=binary)


def test_beam():
    freq_map = RNG.random((3, hp.nside2npix(NSIDE)))
    freq_map_beamed = mock.beam_winpix_correction(NSIDE, freq_map=freq_map, beam_FWHM=100)
    assert np.all(np.isfinite(freq_map_beamed))
    assert freq_map.shape == freq_map_beamed.shape


def test_apply_obsmat():
    freq_map = RNG.random((3, hp.nside2npix(NSIDE)))

    def obsmat_fun(x):
        return x

    freq_map_fun = mock.apply_observation_matrix(obsmat_fun, freq_map)
    assert np.all(freq_map_fun == freq_map)
