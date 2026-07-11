import healpy as hp
import numpy as np
import pytest
from pixell import enmap, utils

from megatop.landscapes import CARLandscape
from megatop.utils import preproc

NSIDE = 8
RNG = np.random.default_rng()


def _car_landscape():
    shape, wcs = enmap.geometry(
        pos=[[-5 * utils.degree, -5 * utils.degree], [5 * utils.degree, 5 * utils.degree]],
        res=0.5 * utils.degree,
        proj="car",
    )
    return CARLandscape(shape, wcs), shape, wcs


def test_common_nside():
    map_1 = RNG.random(size=(3, hp.nside2npix(8)))
    map_2 = RNG.random(size=(3, hp.nside2npix(16)))
    map_3 = RNG.random(size=(3, hp.nside2npix(32)))
    common_nside_maps = preproc.common_beam_and_nside(
        NSIDE, 1, np.ones(3), [map_1, map_2, map_3], 2 * NSIDE
    )
    assert common_nside_maps.shape == (3, 3, hp.nside2npix(NSIDE))


def test_nside_too_small():
    map_1 = RNG.random(size=(3, hp.nside2npix(8)))
    map_2 = RNG.random(size=(3, hp.nside2npix(4)))
    with pytest.raises(ValueError, match="nside"):
        _ = preproc.common_beam_and_nside(NSIDE, 1, np.ones(2), [map_1, map_2], 2 * NSIDE)


def test_common_beam_car_preserves_geometry():
    landscape, shape, wcs = _car_landscape()
    lmax = 2 * 16
    maps = [landscape.synfast(np.ones((4, lmax + 1)), lmax=lmax, seed=[i]) for i in range(2)]
    out = preproc.common_beam_and_nside(
        landscape.working_nside(lmax), 60.0, [30.0, 17.0], maps, lmax
    )
    # CAR common-beam keeps the input geometry: (nfreq, 3, ny, nx).
    assert out.shape == (2, 3, *shape[-2:])
    assert np.all(np.isfinite(out))


def test_read_input_maps_car(tmp_path):
    landscape, shape, wcs = _car_landscape()
    paths = []
    for i in range(2):
        m = landscape.zeros((3,)) + i
        p = tmp_path / f"map_{i}.fits"
        landscape.write_map(p, m)
        paths.append(p)
    maps = preproc.read_input_maps(paths, landscape)
    assert all(isinstance(m, enmap.ndmap) for m in maps)
    assert maps[0].shape == (3, *shape[-2:])
