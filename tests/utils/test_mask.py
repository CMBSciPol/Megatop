import healpy as hp
import numpy as np
from pixell import enmap, utils

from megatop.utils.mask import apply_binary_mask

NSIDE = 32


def test_apply_binary_mask_healpix_1d():
    maps = np.ones((3, hp.nside2npix(NSIDE)))
    bm = np.ones(hp.nside2npix(NSIDE))
    bm[0] = 0
    apply_binary_mask(maps, bm)
    assert np.all(maps[:, 0] == 0)
    assert np.all(maps[:, 1] == 1)


def test_apply_binary_mask_car_2d():
    shape, wcs = enmap.geometry(
        pos=[[-5 * utils.degree, -5 * utils.degree], [5 * utils.degree, 5 * utils.degree]],
        res=0.5 * utils.degree,
        proj="car",
    )
    maps = enmap.zeros((3, *shape[-2:]), wcs=wcs) + 5.0
    bm = enmap.zeros(shape[-2:], wcs=wcs) + 1.0
    bm[0, 0] = 0
    apply_binary_mask(maps, bm)
    assert maps[0, 0, 0] == 0
    assert maps[0, 1, 1] == 5.0
