import healpy as hp
import numpy as np
import scipy as sp
from pysm3 import Sky, units

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
    return np.array([[Cl_TT, Cl_EE, Cl_BB, Cl_TE, Cl_EE * 0, Cl_EE * 0]])


def generate_map_cmb(Cl_cmb_model, nside: int, cmb_seed: list[int] | int | None = None):
    # TODO write tests
    lmax = 3 * nside

    # Fixing seed if required
    # hp.synfast uses the legacy numpy random number generator
    np.random.seed(cmb_seed)  # noqa: NPY002
    map_CMB = hp.synfast(Cl_cmb_model[0], nside=nside, lmax=lmax, new=True, pixwin=False)

    # Resetting seed
    np.random.seed(None)  # noqa: NPY002

    return np.array(map_CMB)


def generate_map_fgs_pysm(map_sets, nside, sky_model, input_coord="G", output_coord="E"):
    # TODO write tests and check coords
    logger.debug(f"Generating FG maps for {[m.freq_tag for m in map_sets]} GHz")
    sky = Sky(nside=nside, preset_strings=sky_model, output_unit=units.uK_CMB)
    maps_fgs = []
    for map_set in map_sets:
        m = sky.get_emission(map_set.frequency * units.GHz, weights=map_set.weight).value
        if input_coord != output_coord:
            logger.debug(
                f"Rotating {map_set.freq_tag}GHz foreground map from {input_coord} to {output_coord}"
            )
            r = hp.Rotator(coord=[input_coord, output_coord])
            m = r.rotate_map_pixel(m)
        maps_fgs.append(m)
    return np.array(maps_fgs)


def get_full_sky_noise_freq_maps(
    map_sets, noise_config: dict, fsky_binary: float, nside: int, id_sim: int = 0
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
            fsky_binary=fsky_binary,
            nside=nside,
            id_sim=id_sim,
        )
    noise_freq_maps = np.zeros((len(map_sets), 3, hp.nside2npix(nside)))
    for i_map_set, map_set in enumerate(map_sets):
        exp = map_set.exp_tag
        noise_config_exp = noise_config.experiments[exp]
        idx_freq = noise_config_exp.default_bands.index(map_set.freq_tag)
        logger.debug(f"Map {exp}_{map_set.freq_tag} has index {idx_freq}.")
        if noise_config_exp.noise_option == NoiseOption.WHITE:
            noise_freq_maps[i_map_set] = get_noise_map_from_white_noise(
                noise_experiment[exp]["map_white_noise_levels"][idx_freq], nside
            )
        elif noise_config_exp.noise_option == NoiseOption.ONE_OVER_F:
            noise_freq_maps[i_map_set] = get_noise_map_from_noise_spectra(
                noise_experiment[exp]["noise_spectra"][idx_freq], nside
            )
        elif noise_config_exp.noise_option == NoiseOption.NOISELESS:
            noise_freq_maps[i_map_set, :, :] = 1e-10
        elif noise_config_exp.noise_option == NoiseOption.NOISE_MAP:
            noise_freq_maps[i_map_set] = hp.ud_grade(
                noise_experiment[exp]["noise_map"][idx_freq], nside_out=nside
            )
        else:
            msg = f"Noise option {noise_config_exp.noise_option} for {exp} is not implemented"
            logger.error(msg)
            raise RuntimeError(msg)
    return noise_freq_maps


def get_noise_experiment(
    exp: str,
    noise_config_exp: ValidExperimentConfig,
    fsky_binary: float,
    nside: int,
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
                f_sky=fsky_binary, ell_max=3 * nside - 1, delta_ell=1, deconv_beam=False
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
                f_sky=fsky_binary,
                ell_max=3 * nside - 1,
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
            f_sky=fsky_binary, ell_max=3 * nside - 1, delta_ell=1, deconv_beam=False
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


def get_noise_map_from_white_noise(map_white_noise_level: float, nside: int):
    logger.debug(f"Map white noise level (Q,U) {map_white_noise_level} muK-arcmin")
    npix = hp.nside2npix(nside)
    nlev_map = np.array(
        [
            map_white_noise_level / np.sqrt(2),
            map_white_noise_level,
            map_white_noise_level,
        ]
    )[:, np.newaxis] * np.ones((3, npix))
    nlev_map /= hp.nside2resol(nside, arcmin=True)
    rng = np.random.default_rng()
    return rng.normal(np.zeros_like(nlev_map), nlev_map, (3, npix))


def get_noise_map_from_noise_spectra(n_ell, nside: int):
    noise_maps = np.zeros((3, hp.nside2npix(nside)))
    noise_spectra = np.zeros((3, 3 * nside - 1))
    logger.warning(
        "Do not trust the temperature noise spectra (ell_knee and alpha_knee are polarisation ones)"
    )
    noise_spectra[0, 2:] = n_ell / 2
    noise_spectra[1, 2:] = n_ell
    noise_spectra[2, 2:] = n_ell
    noise_maps = hp.synfast(
        (
            noise_spectra[0],
            noise_spectra[1],
            noise_spectra[2],
            np.zeros_like(noise_spectra[2]),
        ),
        new=True,
        pixwin=False,
        nside=nside,
    )
    return noise_maps


def include_hits_noise(noise_maps, nhits_maps, binary_mask):
    logger.debug("Rescaling the noise maps by the hits count")
    mask_indices = np.where(binary_mask == 1)[0]
    if np.any(nhits_maps[..., mask_indices] == 0):
        logger.error("Division by 0 in noise map nhit rescaling.")
        logger.error("The binary mask does not cover all areas where nhits = 0.")
        logger.error(
            "Check the 'mask_handling' parameters; adjusting 'binary_mask_zero_threshold' may help."
        )
        logger.error("Exiting...")
    with np.errstate(divide="raise", invalid="raise"):
        noise_maps[..., mask_indices] /= np.sqrt(nhits_maps[..., np.newaxis, mask_indices])

    return noise_maps


def beam_winpix_correction(nside: int, freq_map, beam_FWHM: float):
    lmax_convolution = 3 * nside  # here lmax seems to play an important role
    logger.info(f"Convolving channel with {beam_FWHM} arcmin beam.")
    alms_T, alms_Q, alms_U = hp.map2alm(freq_map, lmax=lmax_convolution, pol=True)
    Bl_gauss_fwhm = hp.gauss_beam(np.radians(beam_FWHM / 60), lmax=lmax_convolution, pol=True)
    wpix_in = hp.pixwin(
        nside, pol=True, lmax=lmax_convolution
    )  # Pixel window function of input maps

    sm_corr_T = Bl_gauss_fwhm[:, 0] * wpix_in[0]
    sm_corr_P = Bl_gauss_fwhm[:, 1] * wpix_in[1]

    # change beam and wpix
    alm_out_T = hp.almxfl(alms_T, sm_corr_T)
    alm_out_E = hp.almxfl(alms_Q, sm_corr_P)
    alm_out_B = hp.almxfl(alms_U, sm_corr_P)

    # alm-->mapf
    alms_out_T, alms_out_Q, alms_out_U = hp.alm2map(
        [alm_out_T, alm_out_E, alm_out_B],
        nside,
        lmax=lmax_convolution,
        pixwin=False,
        fwhm=0.0,
        pol=True,
    )
    freq_map_beamed = [alms_out_T, alms_out_Q, alms_out_U]
    return np.array(freq_map_beamed)


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
