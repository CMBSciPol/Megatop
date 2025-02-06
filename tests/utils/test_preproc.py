import healpy as hp
import numpy as np
import pytest

from megatop.utils import preproc

NSIDE = 8
RNG = np.random.default_rng()


def test_common_nside():
    map_1 = RNG.random(size=(3, hp.nside2npix(8)))
    map_2 = RNG.random(size=(3, hp.nside2npix(16)))
    map_3 = RNG.random(size=(3, hp.nside2npix(32)))
    common_nside_maps = preproc.common_beam_and_nside(NSIDE, 1, np.ones(3), [map_1, map_2, map_3])
    assert common_nside_maps.shape == (3, 3, hp.nside2npix(NSIDE))


def test_nside_too_small():
    map_1 = RNG.random(size=(3, hp.nside2npix(8)))
    map_2 = RNG.random(size=(3, hp.nside2npix(4)))
    with pytest.raises(ValueError, match="Some of input maps have too small nside."):
        _ = preproc.common_beam_and_nside(NSIDE, 1, np.ones(2), [map_1, map_2])
