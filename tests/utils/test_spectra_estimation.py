import numpy as np
import pymaster as nmt
from pixell import enmap, utils

from megatop.utils import mask, spectra

NSIDE = 8
RNG = np.random.default_rng(0)


def _car_patch():
    shape, wcs = enmap.geometry(
        pos=[[-5 * utils.degree, -5 * utils.degree], [5 * utils.degree, 5 * utils.degree]],
        res=0.5 * utils.degree,
        proj="car",
    )
    return shape, wcs


def test_get_common_beam_wpix_car_is_gaussian_only():
    lmax = 64
    car = spectra.get_common_beam_wpix(30.0, NSIDE, lmax, is_car=True)
    # CAR drops the HEALPix pixel window; the beam is the Gaussian common beam (E pol).
    import healpy as hp

    expected = hp.gauss_beam(np.radians(30.0 / 60.0), lmax=lmax, pol=True)[:, 1]
    assert car.shape == (lmax + 1,)
    assert np.allclose(car, expected)


def test_compute_auto_cross_cl_car_fields():
    shape, wcs = _car_patch()
    lmax = 2 * NSIDE
    # Apodized analysis mask + binary mask on the CAR patch.
    binary = enmap.zeros(shape, wcs) + 1.0
    analysis_mask = mask.get_analysis_mask_car(binary, binary.astype(bool), apod_radius_deg=1.0)

    maps_dict = {
        "CMB": enmap.enmap(RNG.standard_normal((2, *shape[-2:])), wcs),
        "Dust": enmap.enmap(RNG.standard_normal((2, *shape[-2:])), wcs),
    }
    beam = spectra.get_common_beam_wpix(30.0, NSIDE, lmax, is_car=True)
    nmt_bins = nmt.NmtBin.from_lmax_linear(lmax, 4)
    workspace = spectra.initialize_nmt_workspace(
        nmt_bins=nmt_bins,
        analysis_mask=analysis_mask,
        beam=beam,
        purify_e=False,
        purify_b=False,
        n_iter=0,
        lmax=lmax,
        wcs=wcs,
    )
    cls = spectra.compute_auto_cross_cl_from_maps_dict(
        maps_dict=maps_dict,
        analysis_mask=analysis_mask,
        workspace=workspace,
        beam=beam,
        n_iter=0,
        lmax=lmax,
        purify_b=False,
        purify_e=False,
        wcs=wcs,
    )
    assert set(cls) == {"CMBxCMB", "CMBxDust", "DustxDust"}
    for v in cls.values():
        assert np.all(np.isfinite(v))
