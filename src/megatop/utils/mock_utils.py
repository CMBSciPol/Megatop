import warnings

import healpy as hp
import numpy as np

import megatop.utils.V3calc as V3


def get_Cl_CMB_model_from_meta(meta):
    """
    This function reads the fiducial CMB Cls from the metadata manager and combines scalar, lensing and tensor
    contributions to return the model Cls according to A_lens and r in the simulation parameter file.

    Parameters
    ----------
        meta: metadata_manager object containing all the config file options

    Returns
    -------
        Cl_cmb_model (ndarray): The model CMB Cls, with shape (num_freq, num_spectra [TT,EE,BB,TE,EB,TB], num_ell).
    """
    path_Cl_BB_lens = meta.get_fname_cls_fiducial_cmb("lensed")
    path_Cl_BB_prim_r1 = meta.get_fname_cls_fiducial_cmb("unlensed_scalar_tensor_r1")
    meta.logger.debug(f"Lensing B-mode path: {path_Cl_BB_lens}")
    meta.logger.debug(f"Primordial B-mode (r=1): {path_Cl_BB_prim_r1}")

    if meta.map_sim_pars is not None:
        r_input = meta.map_sim_pars["r_input"]
        A_lens = meta.map_sim_pars["A_lens_input"]
    else:
        r_input = 0.0
        A_lens = 1.0
    meta.logger.info(f"CMB simulation has r={r_input} and A_lens={A_lens}")
    Cl_BB_prim = r_input * hp.read_cl(path_Cl_BB_prim_r1)[2]
    Cl_lens = hp.read_cl(path_Cl_BB_lens)

    l_max_lens = len(Cl_lens[0])
    Cl_BB_lens = A_lens * Cl_lens[2]
    Cl_TT = Cl_lens[0]
    Cl_EE = Cl_lens[1]
    Cl_TE = Cl_lens[3]

    Cl_BB = Cl_BB_prim[:l_max_lens] + Cl_BB_lens
    Cl_cmb_model = np.array(
        [[Cl_TT, Cl_EE, Cl_BB, Cl_TE, Cl_EE * 0.0, Cl_EE * 0.0]]
    )  # setting TB and EB correlations to 0
    return Cl_cmb_model


def generate_map_cmb(meta, Cl_cmb_model):
    """
    This function generates a CMB map from a Cl_cmb_model array.

    Parameters
    ----------
    meta: metadata_manager
        object containing all the config file options
    Cl_cmb_model :ndarray:
        The model CMB Cls, with shape (num_freq, num_spectra [TT,EE,BB,TE,EB,TB], num_ell).
    Returns
    -------
    map_CMB: ndarray
        CMB map, with shape (3 [T,Q,U], npix).

    Notes
    -----
    * If option fixed_cmb is set to true, the random seed is set to `1234' before synfast at reset to None after.
    """
    lmax = 2 * meta.nside
    if meta.map_sim_pars["fixed_cmb"]:
        np.random.seed(1234)  # Fixing seed so that the CMB is the same for all sims.
    map_CMB = hp.synfast(Cl_cmb_model[0], nside=meta.nside, lmax=lmax, new=True, pixwin=False)
    if meta.map_sim_pars["fixed_cmb"]:
        np.random.seed(None)  # Resetting seed.
    return np.array(map_CMB)


def generate_map_fgs_pysm(meta, input_coord="G", output_coord="E"):
    meta.logger.info(f"Generating FG maps for {(*meta.frequencies,)}GHz")
    from pysm3 import Sky, units

    sky = Sky(nside=meta.nside, preset_strings=meta.sky_model)
    maps_fgs = []
    for _, fr in enumerate(meta.frequencies):
        m = (
            sky.get_emission(fr * units.GHz)
            .to(units.uK_CMB, equivalencies=units.cmb_equivalencies(fr * units.GHz))
            .value
        )
        if input_coord != output_coord:
            meta.logger.info(
                f"Rotating {fr}GHz foreground map from {input_coord} to {output_coord}"
            )
            r = hp.Rotator(coord=[input_coord, output_coord])
            m = r.rotate_map_pixel(m)
        maps_fgs.append(m)
    return np.array(maps_fgs)


def get_noise(meta, fsky_binary):
    """
    This function returns the noise (spectra or map levels) depending on the noise_sim_pars settings in the config file.

    Parameters
    ----------
    meta: metadata_manager
        object containing all the config file options
    fsky_binary: float
        The sky fraction observed.
    Returns
    -------
    n_ell: ndarray or None
        The noise spectra (if computed) shape is (num_freq, num_ell)
    map_white_noise_level: list
        The (polarisation) white noise levels, shape is (num_freqs)
    """
    SO_FREQS = [27, 39, 93, 145, 220, 280]  # Set to match V3 calc
    if meta.noise_sim_pars["experiment"] == "SO":
        meta.logger.info("Using SO V3calc to get white noise levels.")
        idx_freqs = meta.idx_from_list(SO_FREQS)
        _, n_ell, map_white_noise_levels = V3.so_V3_SA_noise(
            sensitivity_mode=meta.noise_sim_pars["sensitivity_mode"],
            one_over_f_mode=meta.noise_sim_pars["knee_mode"],
            SAC_yrs_LF=meta.noise_sim_pars["SAC_yrs_LF"],
            f_sky=fsky_binary,
            ell_max=3 * meta.nside - 1,
            delta_ell=1,
            beam_corrected=False,
            remove_kluge=False,
        )
        map_white_noise_levels = map_white_noise_levels[idx_freqs]
        n_ell = n_ell[idx_freqs]
        meta.logger.info(
            f"Map white noise level (Q,U) {', '.join(f'{level:.2f}' for level in map_white_noise_levels)} muK-arcmin"
        )
        return n_ell, map_white_noise_levels
    else:
        meta.logger.error(
            "NO other options yet"
        )  # shouldn't enter this as checks are done in metadata_manager.


def get_noise_map_from_white_noise(meta, map_white_noise_levels):
    """
    This function returns white noise maps from white noise levels.

    Parameters
    ----------
    meta: metadata_manager
        object containing all the config file options
    map_white_noise_level: list
        The (polarisation) white noise levels, shape is (num_freqs)
    Returns
    -------
    noise_maps: ndarray
        The white noise maps, shape is (num_freqs, 3 [T, Q, U], num_pix)

    Notes
    -----
    * If option include_nhits is True, the noise maps are rescaled by the hit counts.
    """
    nlev_map = np.zeros((len(meta.frequencies), 3, hp.nside2npix(meta.nside)))
    for i_f, _ in enumerate(meta.frequencies):
        nlev_map[i_f] = np.array(
            [
                map_white_noise_levels[i_f] / np.sqrt(2),
                map_white_noise_levels[i_f],
                map_white_noise_levels[i_f],
            ]
        )[:, np.newaxis] * np.ones((3, hp.nside2npix(meta.nside)))
    nlev_map /= hp.nside2resol(meta.nside, arcmin=True)
    noise_maps = np.random.normal(
        np.zeros_like(nlev_map), nlev_map, (len(meta.frequencies), 3, hp.nside2npix(meta.nside))
    )
    if meta.noise_sim_pars["include_nhits"]:
        noise_maps = include_hits_noise(meta, noise_maps)
    else:
        noise_maps = include_hits_noise(meta, noise_maps, binary_only=True)
    return noise_maps


def get_noise_map_from_noise_spectra(meta, n_ell):
    """
    This function returns noise maps from noise power spectra.

    Parameters
    ----------
    meta: metadata_manager
        object containing all the config file options
    n_ell: ndarray
        The noise spectra, shape is (num_freq, num_ell)
    Returns
    -------
    noise_maps: ndarray
        The noise maps, shape is (num_freqs, 3 [T, Q, U], num_pix)

    Notes
    -----
    * If option include_nhits is True, the noise maps are rescaled by the hit counts.
    """
    meta.logger.warning("NOT TESTED YET !!!!")  # TODO TEST THIS !!!!
    noise_maps = np.zeros((len(meta.frequencies), 3, hp.nside2npix(meta.nside)))
    noise_spectra = np.zeros((len(meta.frequencies), 3, 3 * meta.nside - 1))
    noise_spectra[:, 0, 2:] = n_ell / 2
    noise_spectra[:, 1, 2:] = n_ell
    noise_spectra[:, 2, 2:] = n_ell
    for i_f, _ in enumerate(meta.frequencies):
        noise_maps[i_f] = hp.synfast(
            (
                noise_spectra[i_f, 0],
                noise_spectra[i_f, 1],
                noise_spectra[i_f, 2],
                np.zeros_like(noise_spectra[i_f, 2]),
            ),
            new=True,
            pixwin=False,
            nside=meta.nside,
        )
    if meta.noise_sim_pars["include_nhits"]:
        noise_maps = include_hits_noise(meta, noise_maps)
    else:
        noise_maps = include_hits_noise(meta, noise_maps, binary_only=True)
    return noise_maps


def include_hits_noise(meta, noise_maps, unseen=False, binary_only=False):
    """
    This function rescales the noise maps by the hit count map.

    Parameters
    ----------
    meta: metadata_manager
        object containing all the config file options
    noise_maps: ndarray
        The input noise maps, shape is (num_freqs, 3 [T, Q, U], num_pix)
    unseen: bool, optional
        If True, the pixels outside the binary masks are set to hp.UNSEEN, else they are set to 0. Default is False.
    binary_only: bool, optional
        If True, only apply the binary mask (no hit counts). Default is False.
    Returns
    -------
    noise_maps: ndarray
        The noise maps rescaled by the hit counts, shape is (num_freqs, 3 [T, Q, U], num_pix)
    Notes
    -----
    * The TQU maps are rescalded by the same hit count map.
    """
    binary_mask = meta.read_mask("binary")
    if not binary_only:
        meta.logger.info("Rescaling the noise maps by the hits count")
        nhits_map = meta.read_hitmap()
        nhits_map_rescaled = nhits_map / max(nhits_map)
        warnings.filterwarnings("error")
        try:
            noise_maps[..., np.where(binary_mask == 1)[0]] /= np.sqrt(
                nhits_map_rescaled[np.where(binary_mask == 1)[0]]
            )
            # This avoids dividing by 0 in the noise maps
        except RuntimeWarning:
            meta.logger.error("Division by 0 in noise map nhit rescaling.")
            meta.logger.error(
                "This means the binary mask is not covering all the parts where nhits = 0."
            )
            meta.logger.error(
                "Please check the mask_handling parameters; changing 'binary_mask_zero_threshold' can help."
            )
            meta.logger.error("Exiting...")
            exit(1)
        warnings.resetwarnings()
    if unseen:
        noise_maps[..., np.where(binary_mask == 0)[0]] = hp.UNSEEN
    else:
        noise_maps[..., np.where(binary_mask == 0)[0]] = 0.0
    return noise_maps


def beam_winpix_correction(meta, freq_map, beam_FWHM):
    lmax_convolution = 3 * meta.nside  # here lmax seems to play an important role
    meta.logger.info(f"Convolving channel with {beam_FWHM} arcmin beam.")
    alms_T, alms_Q, alms_U = hp.map2alm(freq_map, lmax=lmax_convolution, pol=True)
    Bl_gauss_fwhm = hp.gauss_beam(np.radians(beam_FWHM / 60), lmax=lmax_convolution, pol=True)
    wpix_in = hp.pixwin(
        meta.nside, pol=True, lmax=lmax_convolution
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
        meta.nside,
        lmax=lmax_convolution,
        pixwin=False,
        fwhm=0.0,
        pol=True,
    )
    freq_map_beamed = [alms_out_T, alms_out_Q, alms_out_U]
    return np.array(freq_map_beamed)
