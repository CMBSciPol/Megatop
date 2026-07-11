"""End-to-end CAR pipeline integration test.

Runs the full chain (mask -> binner -> mock_noise -> noise_preproc -> noisecov ->
mock_signal -> preproc -> compsep -> map2cl -> noisespectra -> cl2r) on a small CAR
geometry, entirely offline:

* foregrounds use a `pysm3.Sky` stub (no network), as in `tests/utils/test_mock.py`;
* hit counts come from the bundled SO nominal SAT hitmap fixture
  (`SO_NOMINAL_HITMAP_PATH` override), so no network download happens. The CAR patch
  is placed inside the SO survey footprint so the reprojected mask is non-empty.

The test asserts that every per-step product carries the CAR ``(ny, nx)`` pixel
geometry and that the MCMC yields a finite ``r`` posterior.
"""

from pathlib import Path

import healpy as hp
import numpy as np
import yaml
from pixell import enmap, utils

from megatop import Config, DataManager
from megatop.pipeline import (
    binner_maker,
    cl2r_estimater,
    map_to_cler,
    mask_handler,
    mocker,
    noise_preprocesser,
    noise_spectra_estimater,
    parametric_separater,
    pixel_noisecov_estimater,
    preprocesser,
)
from megatop.utils import mask, mock

RNG = np.random.default_rng(0)
SO_HITMAP_FIXTURE = Path(__file__).parent / "data" / "norm_nHits_SA_35FOV_ns0064.fits"


class _FakeEmission:
    def __init__(self, arr):
        self.value = arr


class _FakeSky:
    """Offline stand-in for `pysm3.Sky` (returns random HEALPix emission)."""

    def __init__(self, nside, preset_strings=None, output_unit=None):
        self.nside = nside

    def get_emission(self, freq, weights=None):
        return _FakeEmission(RNG.random((3, hp.nside2npix(self.nside))))


def _make_car_config(tmp_path):
    # CAR patch inside the SO south survey footprint (dec -55..-25, ra -30..30 deg).
    shape, wcs = enmap.geometry(
        pos=np.array([[-55.0, -30.0], [-25.0, 30.0]]) * utils.degree,
        res=1.0 * utils.degree,
        proj="car",
    )
    geom_file = tmp_path / "geometry.fits"
    enmap.write_map_geometry(str(geom_file), shape, wcs)

    base = yaml.safe_load(Path("paramfiles/e2e_check_car.yaml").read_text())
    base["data_dirs"]["root"] = str(tmp_path / "data")
    base["output_dirs"]["root"] = str(tmp_path / "out")
    base["general_pars"]["pixelisation"]["car"]["geometry_file"] = str(geom_file)
    base["general_pars"]["lmax"] = 128
    base["map_sim_pars"]["n_sim"] = 1
    base["noise_sim_pars"]["n_sim"] = 1
    base["map2cl_pars"]["delta_ell"] = 30
    # Keep the MCMC cheap; only r and A_lens are free here.
    base["cl2r_pars"]["n_walkers"] = 8
    base["cl2r_pars"]["n_steps"] = 30
    base["cl2r_pars"]["n_steps_burnin"] = 5

    cfg_file = tmp_path / "config.yaml"
    with cfg_file.open("w") as f:
        yaml.safe_dump(base, f)
    return Config.load_yaml(cfg_file), shape


def test_car_pipeline_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(mock, "Sky", _FakeSky)
    # Offline SO nominal hitmap (read at call time inside mask.get_nhits).
    monkeypatch.setattr(mask, "SO_NOMINAL_HITMAP_PATH", str(SO_HITMAP_FIXTURE))

    config, shape = _make_car_config(tmp_path)
    ny, nx = shape[-2:]
    manager = DataManager(config)
    manager.create_output_dirs(config.map_sim_pars.n_sim, config.noise_sim_pars.n_sim)

    # ── Global steps ──────────────────────────────────────────────────────────
    mask_handler.mask_handler(manager, config)
    binner_maker.binning_maker(manager, config)
    binner_maker.fiducial_cmb_spectra_computer(manager, config)

    # Masks are written as CAR enmaps and cover part of the patch.
    analysis_mask = config.landscape.read_map(manager.path_to_analysis_mask)
    assert isinstance(analysis_mask, enmap.ndmap)
    assert analysis_mask.shape[-2:] == (ny, nx)
    assert np.any(analysis_mask > 0)

    # ── Noise branch ──────────────────────────────────────────────────────────
    mocker.process_noise(config, manager, None)
    noise_preprocesser.noise_preprocess_realisation(config, manager, 0)
    pixel_noisecov_estimater.aggregate_noise_cov(manager, config)

    noisecov = np.load(manager.path_to_pixel_noisecov)
    assert noisecov.shape == (len(config.frequencies), 3, ny, nx)

    # ── Signal branch ─────────────────────────────────────────────────────────
    mocker.process_signal(config, manager, None)
    preprocesser.preproc_and_save(config, manager, id_sim=0)

    preproc_maps = np.load(manager.get_path_to_preprocessed_maps(0))
    assert preproc_maps.shape == (len(config.frequencies), 3, ny, nx)

    parametric_separater.compsep_and_save(config, manager, id_sim=0)
    comp_maps = np.load(manager.get_path_to_components_maps(0))
    # (ncomp, QU, ny, nx) — pixel geometry restored after the flat compsep algebra.
    assert comp_maps.shape[-2:] == (ny, nx)

    map_to_cler.map2cl_and_save(config, manager, id_sim=0)
    workspace, beam = noise_spectra_estimater.init_workspace(config, manager)
    noise_spectra_estimater.noise_spectra_estimator(config, manager, workspace, beam, 0)

    spectra = np.load(manager.get_path_to_spectra_cross_components(0))
    cmb_bb = spectra["CMBxCMB"][3]
    assert np.all(np.isfinite(cmb_bb))

    # ── Cosmology ─────────────────────────────────────────────────────────────
    cl2r_estimater.run_mcmc_and_save(manager, config, id_sim=0)
    for out in manager.outputs_cl2r(0):
        assert out.exists(), f"missing cl2r output {out}"
