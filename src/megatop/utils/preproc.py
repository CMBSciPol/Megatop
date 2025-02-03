import copy
import os

import healpy as hp
import numpy as np

from ..config import Config
from ..data_manager import DataManager
from .logger import logger
from .timer import function_timer


def apply_binary_mask(meta, freq_maps, unseen=False):
    """
    This function applies the binary mask to the frequency maps.

    Parameters
    ----------
    meta: metadata_manager
        object containing all the config file options
    freq_maps: ndarray
        The frequency maps, with shape (num_freq, num_stokes, num_pixels).
    unseen: bool, optional
        If True, the pixels outside the binary masks are set to hp.UNSEEN, else they are set to 0. Default is False.
    Returns
    -------
    freq_maps_masked: ndarray
        The frequency maps after applying the binary mask, with shape (num_freq, num_stokes, num_pixels).
    Notes
    -----
    * The input maps should all have the same shape.
    """
    meta.timer.start("masking")
    binary_mask = meta.read_mask("binary")
    freq_maps_masked = copy.deepcopy(freq_maps)
    if unseen:
        freq_maps_masked[..., np.where(binary_mask == 0)[0]] = hp.UNSEEN
    else:
        freq_maps_masked[..., np.where(binary_mask == 0)[0]] = 0.0
    meta.timer.stop("masking", "Applying binary mask")
    return freq_maps_masked


@function_timer("apply-binary-mask")
def _apply_binary_mask(manager: DataManager, freq_maps, unseen=False):
    binary_mask = hp.read_map(manager.path_to_binary_mask)
    freq_maps_masked = copy.deepcopy(freq_maps)
    if unseen:
        freq_maps_masked[..., np.where(binary_mask == 0)[0]] = hp.UNSEEN
    else:
        freq_maps_masked[..., np.where(binary_mask == 0)[0]] = 0.0
    return freq_maps_masked


def common_beam_and_nside(meta, freq_maps):
    """
    This function takes the frequency maps and applies the common beam correction, deconvolves the frequency beams,
    changes the NSIDE of the maps and includes the effect of the pixel window function.

    Parameters
    ----------
    meta: metadata_manager
        object containing all the config file options
    freq_maps: ndarray
        The frequency maps, with shape (num_freq, num_stokes, num_pixels).
    Returns
    -------
    freq_maps_out : ndarray
        The frequency maps after the common beam correction, frequency beams decovolution, NSIDE change,
        and pixel window function effect,
        with shape (num_freq, num_stokes, num_pixels).
    Notes
    -----
    * The input maps are allowed to have different `num_pixels' to allow for different resolutions.
    """

    meta.timer.start("common")
    freq_maps_out = []
    logger.info(
        f"Common beam correction -> {meta.pre_proc_pars['common_beam_correction']} arcmin and NSIDE -> {meta.nside}"
    )
    logger.info(
        "Correcting for frequency-dependent beams, convolving with a common beam, modifying NSIDE and including effect of pixel window function."
    )
    lmax_convolution = 3 * meta.nside
    wpix_out = hp.pixwin(
        meta.nside, pol=True, lmax=lmax_convolution
    )  # Pixel window function of output maps
    Bl_gauss_common = hp.gauss_beam(
        np.radians(meta.pre_proc_pars["common_beam_correction"] / 60),
        lmax=lmax_convolution,
        pol=True,
    )

    for i_f, freq in enumerate(meta.frequencies):
        # Input window function. WARNING: Must be done in the loop, as input map don't necessarily have the same nside (e.g. MSS2)
        logger.info(
            f"Pre-processing {freq} GHz (input beam FWHM {meta.beams_FWHM_arcmin[i_f]} arcmin)"
        )
        wpix_in = hp.pixwin(
            hp.get_nside(freq_maps[i_f, 0]), pol=True, lmax=lmax_convolution
        )  # Pixel window function of input maps
        wpix_in[1][0:2] = 1.0  # in order not to divide by 0

        # beam corrections
        Bl_gauss_fwhm = hp.gauss_beam(
            np.radians(meta.beams_FWHM_arcmin[i_f] / 60), lmax=lmax_convolution, pol=True
        )

        bl_correction = Bl_gauss_common / Bl_gauss_fwhm

        sm_corr_T = bl_correction[:, 0] * wpix_out[0] / wpix_in[0]
        sm_corr_P = bl_correction[:, 1] * wpix_out[1] / wpix_in[1]

        # map-->alm
        map_in_T = np.array(freq_maps[i_f, 0])
        map_in_Q = np.array(freq_maps[i_f, 1])
        map_in_U = np.array(freq_maps[i_f, 2])

        alm_in_T, alm_in_E, alm_in_B = hp.map2alm(
            [map_in_T, map_in_Q, map_in_U], lmax=lmax_convolution, pol=True, iter=10
        )
        # here lmax seems to play an important role

        # change beam and wpix
        alm_out_T = hp.almxfl(alm_in_T, sm_corr_T)
        alm_out_E = hp.almxfl(alm_in_E, sm_corr_P)
        alm_out_B = hp.almxfl(alm_in_B, sm_corr_P)

        # alm-->mapf
        map_out_T, map_out_Q, map_out_U = hp.alm2map(
            [alm_out_T, alm_out_E, alm_out_B],
            meta.general_pars["nside"],
            lmax=lmax_convolution,
            pixwin=False,
            fwhm=0.0,
            pol=True,
        )

        # a priori all the options are set to there default, even lmax which is computed wrt input alms
        out_map = np.array([map_out_T, map_out_Q, map_out_U])
        freq_maps_out.append(out_map)
    meta.timer.stop("common", "Common beam and nside")
    return np.array(freq_maps_out)


@function_timer("common-beam-and-nside")
def _common_beam_and_nside(config: Config, freq_maps):
    freq_maps_out = []
    logger.info(
        f"Common beam correction -> {config.pre_proc_pars.common_beam_correction} arcmin and NSIDE -> {config.nside}"
    )
    logger.info(
        "Correcting for frequency-dependent beams, convolving with a common beam, modifying NSIDE and including effect of pixel window function."
    )
    lmax_convolution = 3 * config.nside
    wpix_out = hp.pixwin(
        config.nside, pol=True, lmax=lmax_convolution
    )  # Pixel window function of output maps
    Bl_gauss_common = hp.gauss_beam(
        np.radians(config.pre_proc_pars.common_beam_correction / 60),
        lmax=lmax_convolution,
        pol=True,
    )

    for i_f, freq in enumerate(config.frequencies):
        # Input window function. WARNING: Must be done in the loop, as input map don't necessarily have the same nside (e.g. MSS2)
        logger.info(f"Pre-processing {freq} GHz (input beam FWHM {config.beams[i_f]} arcmin)")
        wpix_in = hp.pixwin(
            hp.get_nside(freq_maps[i_f, 0]), pol=True, lmax=lmax_convolution
        )  # Pixel window function of input maps
        wpix_in[1][0:2] = 1.0  # in order not to divide by 0

        # beam corrections
        Bl_gauss_fwhm = hp.gauss_beam(
            np.radians(config.beams[i_f] / 60), lmax=lmax_convolution, pol=True
        )

        bl_correction = Bl_gauss_common / Bl_gauss_fwhm

        sm_corr_T = bl_correction[:, 0] * wpix_out[0] / wpix_in[0]
        sm_corr_P = bl_correction[:, 1] * wpix_out[1] / wpix_in[1]

        # map-->alm
        map_in_T = np.array(freq_maps[i_f, 0])
        map_in_Q = np.array(freq_maps[i_f, 1])
        map_in_U = np.array(freq_maps[i_f, 2])

        alm_in_T, alm_in_E, alm_in_B = hp.map2alm(
            [map_in_T, map_in_Q, map_in_U], lmax=lmax_convolution, pol=True, iter=10
        )
        # here lmax seems to play an important role

        # change beam and wpix
        alm_out_T = hp.almxfl(alm_in_T, sm_corr_T)
        alm_out_E = hp.almxfl(alm_in_E, sm_corr_P)
        alm_out_B = hp.almxfl(alm_in_B, sm_corr_P)

        # alm-->mapf
        map_out_T, map_out_Q, map_out_U = hp.alm2map(
            [alm_out_T, alm_out_E, alm_out_B],
            config.nside,
            lmax=lmax_convolution,
            pixwin=False,
            fwhm=0.0,
            pol=True,
        )

        # a priori all the options are set to there default, even lmax which is computed wrt input alms
        out_map = np.array([map_out_T, map_out_Q, map_out_U])
        freq_maps_out.append(out_map)
    return np.array(freq_maps_out)


def read_input_maps(meta):
    """
    This function reads the frequency maps from the files and returns them as an array.

    Parameters
    ----------
    meta: metadata_manager
        object containing all the config file options

    Returns
    -------
        combined_maps : (ndarray)
            The frequency maps, with shape (num_freq, num_stokes, num_pixels).
    Notes
    -----
    * The input maps are allowed to have different `num_pixels' to allow for different resolutions.
    """
    freq_maps_input = []
    for map in meta.maps_list:
        fname = os.path.join(meta.map_directory, meta.map_sets[map]["file_root"] + ".fits")
        logger.debug(f"Reading map from {fname}")
        freq_maps_input.append(hp.read_map(fname, field=None).tolist())
    return np.array(freq_maps_input, dtype=object)


def _read_input_maps(manager: DataManager):
    freq_maps_input = []
    for mapname in manager.get_maps_filenames():
        logger.debug(f"Reading map from {mapname}")
        # TODO: just return a list of arrays with different sizes
        freq_maps_input.append(hp.read_map(mapname, field=None).tolist())
    return np.array(freq_maps_input, dtype=object)
