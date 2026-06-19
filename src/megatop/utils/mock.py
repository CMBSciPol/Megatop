import os
import zlib

import healpy as hp
import numpy as np
import scipy as sp
from pixell import enmap
from pysm3 import Sky, units

import megatop.utils.harmonic as hu
from megatop.landscapes import AbstractLandscape

from ..config import (
    CustomSATConfig,
    ExternalNoiseMapconfig,
    NoiseOption,
    SOConfig,
    ValidExperimentConfig,
)
from ..data_manager import DataManager
from . import V3calc as V3
from . import V3p1calc as V3p1
from .logger import logger

HEALPY_DATA_PATH = os.getenv("HEALPY_LOCAL_DATA", None)


def get_Cl_CMB_model_from_manager(manager: DataManager):
    # TODO make this a method of DataManager
    logger.debug(f"Lensing B-mode path: {manager.path_to_lensed_scalar}")
    logger.debug(f"Primordial B-mode (r=1): {manager.path_to_unlensed_scalar_tensor_r1}")

    r_input = manager._config.map_sim_pars.r_input
    A_lens = manager._config.map_sim_pars.A_lens
    logger.debug(f"CMB simulation has r={r_input} and A_lens={A_lens}")
    Cl_lens = hp.read_cl(manager.path_to_lensed_scalar)
    Cl_BB_prim = r_input * hp.read_cl(manager.path_to_unlensed_scalar_tensor_r1)[2]

    l_max_lens = len(Cl_lens[0])
    Cl_BB_lens = A_lens * Cl_lens[2]
    Cl_TT = Cl_lens[0]
    Cl_EE = Cl_lens[1]
    Cl_TE = Cl_lens[3]

    Cl_BB = Cl_BB_prim[:l_max_lens] + Cl_BB_lens

    # setting TB and EB correlations to 0
    return np.array([Cl_TT, Cl_EE, Cl_BB, Cl_TE, Cl_EE * 0, Cl_EE * 0])


# Template native nside per PySM model: we want to avoid PySM `ud_grade`ing from native resolution.
# - PySM2 d0-d8 / s0-s3: nside-512 template
# - PySM3 d9, d10, d12, s4, s5, s7: nside-2048 is the minimum available
# - PySM3 d11, s6: small scales synthesized at the requested nside
_PYSM_LOW_RES = {"d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "s0", "s1", "s2", "s3"}
_PYSM_HIGH_RES = {"d9", "d10", "d12", "s4", "s5", "s7"}


def pysm_render_nside(sky_model: list[str], working_nside: int) -> int:
    floor = working_nside
    for model in sky_model:
        if model in _PYSM_LOW_RES:
            floor = max(floor, 512)
        elif model in _PYSM_HIGH_RES:
            floor = max(floor, 2048)
    return floor


def generate_map_fgs_pysm(
    map_sets,
    lmax: int,
    sky_model: list[str],
    landscape: AbstractLandscape,
):
    """Render PySM foregrounds and project them onto the target geometry.

    PySM renders in HEALPix at the template-native nside. The galactic→equatorial
    rotation is done in harmonic space (`rotate_alm`, exact for a band-limited
    field), staying HEALPix at the native nside. The map is then resampled onto
    the target pixelisation in PIXEL space (`reproject_pixel`: HEALPix `ud_grade`,
    CAR spline interpolation). Keeping the final resampling in pixel space avoids
    the Gibbs ringing a harmonic round-trip produces around the bright
    galactic-plane features of the PySM templates.
    """
    pysm_nside = pysm_render_nside(sky_model, landscape.working_nside(lmax))
    logger.debug(
        f"Generating FG maps for {[m.freq_tag for m in map_sets]} GHz at nside {pysm_nside}"
    )
    sky = Sky(nside=pysm_nside, preset_strings=sky_model, output_unit=units.uK_CMB)
    maps_fgs = []
    for map_set in map_sets:
        m = sky.get_emission(map_set.frequency * units.GHz, weights=map_set.weight).value
        logger.debug(f"Projecting {map_set.freq_tag}GHz foreground map (gal->equ)")
        m = hu.rotate_map_alms(m, ["G", "C"], spin=[0, 2])  # galactic -> equatorial
        m = landscape.reproject_pixel(m, spin=(0, 2))
        maps_fgs.append(m)
    return landscape.stack(maps_fgs)


def get_full_sky_noise_freq_maps(
    map_sets,
    noise_config: dict,
    fsky_effective: float,
    landscape,
    lmax: int,
    id_sim: int = 0,
    seed=None,
):
    experiments_map_set = set([map_set.exp_tag for map_set in map_sets])
    experiments_noiseconfig = [name for name in noise_config.experiments]
    noise_experiment = {}
    for exp in experiments_map_set:
        try:
            assert exp in experiments_noiseconfig
        except AssertionError as e:
            msg = f"No noise sim config for {exp}"
            logger.error(msg)
            raise RuntimeError(msg) from e
        noise_experiment[exp] = get_noise_experiment(
            exp,
            noise_config.experiments[exp],
            fsky_effective=fsky_effective,
            lmax=lmax,
            id_sim=id_sim,
        )
    noise_freq_maps = landscape.zeros((len(map_sets), 3))
    for i_map_set, map_set in enumerate(map_sets):
        exp = map_set.exp_tag
        noise_config_exp = noise_config.experiments[exp]
        idx_freq = noise_config_exp.default_bands.index(map_set.freq_tag)
        logger.debug(f"Map {exp}_{map_set.freq_tag} has index {idx_freq}.")
        seed_i = list(seed) + [zlib.crc32(map_set.name.encode())] if seed is not None else None
        logger.debug(f"Noise {seed_i = } for {map_set.name}")
        if noise_config_exp.noise_option == NoiseOption.WHITE:
            noise_freq_maps[i_map_set] = get_noise_map_from_white_noise(
                noise_experiment[exp]["map_white_noise_levels"][idx_freq],
                landscape,
                seed=seed_i,
            )
        elif noise_config_exp.noise_option == NoiseOption.ONE_OVER_F:
            noise_freq_maps[i_map_set] = get_noise_map_from_noise_spectra(
                noise_experiment[exp]["noise_spectra"][idx_freq],
                lmax,
                landscape,
                seed=seed_i,
            )
        elif noise_config_exp.noise_option == NoiseOption.NOISELESS:
            noise_freq_maps[i_map_set, :, :] = 1e-10
        elif noise_config_exp.noise_option == NoiseOption.NOISE_MAP:
            external = noise_experiment[exp]["noise_map"][idx_freq]
            # external maps are HEALPix; resample/reproject onto the target geometry
            noise_freq_maps[i_map_set] = landscape.reproject_pixel(external, rot=None)
        else:
            msg = f"Noise option {noise_config_exp.noise_option} for {exp} is not implemented"
            logger.error(msg)
            raise RuntimeError(msg)
    return noise_freq_maps


def get_noise_experiment(
    exp: str,
    noise_config_exp: ValidExperimentConfig,
    fsky_effective: float,
    lmax: int,
    id_sim: int = 0,
):
    if type(noise_config_exp) is SOConfig:
        if noise_config_exp.usev3p1:
            logger.info(
                f"Getting noise model ({noise_config_exp.noise_option}) for {exp} using V3p1 calc"
            )
            nc = V3p1.SOSatV3point1(
                sensitivity_mode=noise_config_exp.v3_sensitivity_mode,
                N_tubes=noise_config_exp.Ntubes_years,
                one_over_f_mode=noise_config_exp.v3_one_over_f_mode,
                survey_years=1.0,  # The scaling wiht time is done through Ntubes_years
            )
            _, _, n_ell, white_noise_levels = nc.get_noise_curves(
                f_sky=fsky_effective, ell_max=lmax + 1, delta_ell=1, deconv_beam=False
            )
        else:
            logger.info(
                f"Getting noise model ({noise_config_exp.noise_option}) for {exp} using V3 calc"
            )
            sensitivity_mode = noise_config_exp.v3_sensitivity_mode
            one_over_f_mode = noise_config_exp.v3_one_over_f_mode
            _, n_ell, white_noise_levels = V3.so_V3_SA_noise(
                sensitivity_mode=sensitivity_mode,
                one_over_f_mode=one_over_f_mode,
                SAC_yrs_LF=noise_config_exp.SAC_yrs_LF,
                f_sky=fsky_effective,
                ell_max=lmax + 1,
                delta_ell=1,
                beam_corrected=False,
                remove_kluge=False,
            )

    elif type(noise_config_exp) is CustomSATConfig:
        logger.info(
            f"Getting noise model ({noise_config_exp.noise_option}) for {exp} using v3p1 calc with customSAT"
        )
        nc = V3p1.CustomSAT(
            bands=noise_config_exp.default_bands,
            sensitivities=noise_config_exp.sensitivities,
            N_tubes=noise_config_exp.Ntubes_years,
            ell_knee=noise_config_exp.ell_knee,
            alpha_knee=noise_config_exp.alpha_knee,
            survey_years=1.0,
        )
        _, _, n_ell, white_noise_levels = nc.get_noise_curves(
            f_sky=fsky_effective, ell_max=lmax + 1, delta_ell=1, deconv_beam=False
        )

    elif type(noise_config_exp) is ExternalNoiseMapconfig:
        logger.info(f"Reading noise map from {noise_config_exp.root} for {exp}.")
        fname_list = [
            noise_config_exp.root
            / f"{id_sim:04d}"
            / f"{noise_config_exp.prefix}{int(fr):03d}{noise_config_exp.suffix}.fits"
            for fr in noise_config_exp.default_bands
        ]  # FIXED FILE EXTENSION
        external_map_list = [
            noise_config_exp.correction * hp.read_map(fname) for fname in fname_list
        ]
        return {"noise_map": external_map_list}

    else:
        msg = f"Noise config {type(noise_config_exp)} for {exp} is not recognized"
        logger.error(msg)
        raise RuntimeError(msg)

    return {"noise_spectra": n_ell, "map_white_noise_levels": white_noise_levels}


def get_noise_map_from_white_noise(depth_qu: float, landscape, seed=None):
    logger.debug(f"Map white noise level (Q,U) {depth_qu} muK-arcmin")
    # per-Stokes noise level (T = P/sqrt(2)) in muK-arcmin
    stokes_level = np.array([depth_qu / np.sqrt(2), depth_qu, depth_qu])
    # sigma per pixel = level / sqrt(pixel area); pixel area is a scalar for HEALPix
    # and a (ny, nx) enmap for CAR (varies with declination)
    sqrt_area = np.sqrt(landscape.pixel_area_arcmin2())
    # For CAR sqrt_area is an enmap so the result carries the wcs.
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((3, *landscape.pixel_shape)) / sqrt_area
    # Double transpose broadcasts stokes_level over the leading Stokes axis.
    return (noise.T * stokes_level).T


def get_noise_map_from_noise_spectra(n_ell, lmax: int, landscape, seed=None):
    noise_spectra = np.zeros((3, lmax + 1))
    logger.warning(
        "Do not trust the temperature noise spectra (ell_knee and alpha_knee are polarisation ones)"
    )
    noise_spectra[0, 2:] = n_ell / 2
    noise_spectra[1, 2:] = n_ell
    noise_spectra[2, 2:] = n_ell
    cl = np.array(
        [noise_spectra[0], noise_spectra[1], noise_spectra[2], np.zeros_like(noise_spectra[2])]
    )
    return landscape.synfast(cl, lmax=lmax, seed=seed)


def include_hits_noise(noise_maps, common_nhits_map, binary_mask):
    logger.debug("Rescaling the noise maps by the hits count")
    # boolean mask works for both HEALPix (npix,) and CAR (ny, nx) pixel axes
    good = np.asarray(binary_mask) == 1
    if np.any(np.asarray(common_nhits_map)[good] == 0):
        logger.error("Division by 0 in noise map nhit rescaling.")
        logger.error("The binary mask does not cover all areas where nhits = 0.")
        logger.error(
            "Check the 'mask_handling' parameters; adjusting 'binary_mask_zero_threshold' may help."
        )
        logger.error("Exiting...")
    with np.errstate(divide="raise", invalid="raise"):
        noise_maps[..., good] /= np.sqrt(np.asarray(common_nhits_map)[good])

    return noise_maps


def beam_winpix_correction(nside: int, freq_map, beam_FWHM: float, lmax: int):
    # here lmax seems to play an important role
    logger.info(f"Convolving channel with {beam_FWHM} arcmin beam.")
    # geometry comes from the input map: an enmap means CAR, ndarray means HEALPix
    car = isinstance(freq_map, enmap.ndmap)
    alms_in = hu.map2alm(freq_map, spin=[0, 2], lmax=lmax)
    Bl_gauss_fwhm = hu.gauss_beam(beam_FWHM, lmax, pol=True)

    if car:
        # apply the (Gaussian) beam in harmonic space, then the CAR pixel window
        # in map space via enmap.apply_window
        hu.almxfl(alms_in[0], Bl_gauss_fwhm[:, 0], inplace=True)
        hu.almxfl(alms_in[1:], Bl_gauss_fwhm[:, 1], inplace=True)
        out = hu.alm2map(alms_in, spin=[0, 2], shape=freq_map.shape, wcs=freq_map.wcs, lmax=lmax)
        return enmap.apply_window(out, pow=1)

    wpix_in = hp.pixwin(
        nside,
        pol=True,
        lmax=lmax,
        datapath=HEALPY_DATA_PATH,
    )  # Pixel window function of input maps

    sm_corr_T = Bl_gauss_fwhm[:, 0] * wpix_in[0]
    sm_corr_P = Bl_gauss_fwhm[:, 1] * wpix_in[1]

    # change beam and wpix
    hu.almxfl(alms_in[0], sm_corr_T, inplace=True)
    hu.almxfl(alms_in[1:], sm_corr_P, inplace=True)

    return hu.alm2map(alms_in, spin=[0, 2], nside=nside, lmax=lmax)


def load_observation_matrix(nside: int, map_sets, obsmat_filenames) -> dict:
    dict_obsmats_func = {}
    for map_set, fname in zip(map_sets, obsmat_filenames, strict=False):
        logger.info(f"Loading obsmat for {map_set.name} from {fname}")
        obsmat = sp.sparse.load_npz(fname)
        dict_obsmats_func[map_set.name] = lambda map_, obs_mat=obsmat: obs_mat.dot(
            map_.ravel()
        ).reshape(3, hp.nside2npix(nside))
    return dict_obsmats_func


def apply_observation_matrix(obsmat_func, freq_map):
    return hp.reorder(obsmat_func(hp.reorder(freq_map, r2n=True)), n2r=True)
