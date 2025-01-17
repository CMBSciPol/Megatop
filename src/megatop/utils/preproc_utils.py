import os
import healpy as hp
import numpy as np
import copy
import os

import healpy as hp
import numpy as np


def ApplyBinaryMask(meta, freq_maps, use_UNSEEN=False):
    """
    This function applies the binary mask to the frequency maps.

    Args:
        meta: The metadata manager.
        freq_maps (ndarray): The frequency maps to mask, with shape (num_freq, num_stokes, num_pixels).
        use_UNSEEN (bool): If True, the UNSEEN value is used for the masked pixels, otherwise the masked pixels are set to zero.

    Returns:
        freq_maps_masked (ndarray): The frequency maps after applying the binary mask, with shape (num_freq, num_stokes, num_pixels).
    """
    
    meta.timer.start("mask")
    binary_mask_path = meta.get_fname_mask("binary")
    binary_mask = hp.read_map(binary_mask_path, dtype=float)

    freq_maps_masked = copy.deepcopy(freq_maps)

    if use_UNSEEN:
        freq_maps_masked[:, np.where(binary_mask == 0)[0]] = hp.UNSEEN
    else:
        freq_maps_masked *= binary_mask
    meta.timer.stop("mask", meta.logger,"Masking")
    return freq_maps_masked


def save_preprocessed_maps(meta, freq_maps_beamed_masked):
    """
    This function saves the pre-processed maps (masked and beamed) to a .npy file in the pre-processing directory with name
    "freq_maps_preprocessed.npy".

    Args:
        meta: The metadata manager.
        freq_maps_beamed_masked (ndarray): The frequency maps after applying the binary mask and the common beam correction, with shape (num_freq, num_stokes, num_pixels).

    Returns:
        None
    """

    fname = os.path.join(meta.pre_process_directory, "freq_maps_preprocessed.npy")
    np.save(fname, freq_maps_beamed_masked)
    meta.logger.info(f"Pre-processed maps (masked and beamed) saved to {fname}")

def CommonBeamConvAndNsideModification(meta, freq_maps):
    """
    This function takes the frequency maps and applies the common beam correction, deconvolves the frequency beams,
    changes the NSIDE of the maps and includes the effect of the pixel window function.

    Args:
        meta: The metadata manager.
        freq_maps (ndarray): The frequency maps, with shape (num_freq, num_stokes, num_pixels).

    Returns:
        freq_maps_out (ndarray): The frequency maps after the common beam correction,
                                 frequency beams decovolution, NSIDE change,
                                 and pixel window function effect,
                                 with shape (num_freq, num_stokes, num_pixels).
    """


    meta.timer.start("beam")

    freq_maps_out = []

    meta.logger.info(
        "-> common beam correction and NSIDE change: correcting for frequency-dependent beams, convolving with a common beam, modifying NSIDE and include effect of pixel window function"
    )

    lmax_convolution = 3 * meta.general_pars["nside"]
    wpix_out = hp.pixwin(
        meta.general_pars["nside"], pol=True, lmax=lmax_convolution
    )  # Pixel window function of output maps
    Bl_gauss_common = hp.gauss_beam(
        np.radians(meta.pre_proc_pars["common_beam_correction"] / 60),
        lmax=lmax_convolution,
        pol=True,
    )

    for f in range(len(meta.frequencies)):
        # Input window function. WARNING: Must be done in the loop, as input map don't necessarily have the same nside (e.g. MSS2)
        wpix_in = hp.pixwin(
            hp.npix2nside(len(freq_maps[f, 0])), pol=True, lmax=lmax_convolution
        )  # Pixel window function of input maps
        wpix_in[1][0:2] = 1.0  # in order not to divide by 0

        # beam corrections
        Bl_gauss_fwhm = hp.gauss_beam(
            np.radians(meta.pre_proc_pars["fwhm"][f] / 60), lmax=lmax_convolution, pol=True
        )

        bl_correction = Bl_gauss_common / Bl_gauss_fwhm

        sm_corr_T = bl_correction[:, 0] * wpix_out[0] / wpix_in[0]
        sm_corr_P = bl_correction[:, 1] * wpix_out[1] / wpix_in[1]

        # map-->alm
        cmb_in_T = np.array(freq_maps[f, 0])
        cmb_in_Q = np.array(freq_maps[f, 1])
        cmb_in_U = np.array(freq_maps[f, 2])

        alm_in_T, alm_in_E, alm_in_B = hp.map2alm(
            [cmb_in_T, cmb_in_Q, cmb_in_U], lmax=lmax_convolution, pol=True, iter=10
        )
        # here lmax seems to play an important role

        # change beam and wpix
        alm_out_T = hp.almxfl(alm_in_T, sm_corr_T)
        alm_out_E = hp.almxfl(alm_in_E, sm_corr_P)
        alm_out_B = hp.almxfl(alm_in_B, sm_corr_P)

        # alm-->mapf
        cmb_out_T, cmb_out_Q, cmb_out_U = hp.alm2map(
            [alm_out_T, alm_out_E, alm_out_B],
            meta.general_pars["nside"],
            lmax=lmax_convolution,
            pixwin=False,
            fwhm=0.0,
            pol=True,
        )

        # a priori all the options are set to there default, even lmax which is computed wrt input alms
        out_map = np.array([cmb_out_T, cmb_out_Q, cmb_out_U])
        freq_maps_out.append(out_map)
    meta.timer.stop("beam", meta.logger, "Common beam convolution")
    return np.array(freq_maps_out)


def read_maps(meta):
    """
    This function reads the frequency maps from the files and returns them as an array.

    Args:
        meta: The metadata manager.

    Returns:
        cmb_fg_freq_maps_beamed (ndarray): The frequency maps, with shape (num_freq, num_stokes, num_pixels).

    """

    cmb_fg_freq_maps_beamed = []
    for i, map in enumerate(meta.maps_list):
        fname = os.path.join(meta.map_directory, meta.map_sets[map]["file_root"] + ".fits")
        cmb_fg_freq_maps_beamed.append(hp.read_map(fname, field=None).tolist())
    return np.array(cmb_fg_freq_maps_beamed, dtype=object)


def read_maps_ben_sims(meta, id_sim=0):
    """
    This function reads the frequency maps from the files of the ben sims and returns them as an array.

    Args:
        meta: The metadata manager.
        id_sim (int): The id of the simulation to read the maps from.

    Returns:
        cmb_fg_freq_maps_beamed (ndarray): The frequency maps, with shape (num_freq, num_stokes, num_pixels).

    """

    meta.logger.info("WARNING: ben_sims related function will be removed")
    # TODO: REMOVE ben_sims related function
    
    if hasattr(meta, "ben_unfiltered") and meta.ben_unfiltered:
        beamed_sky_unfiltered = np.load(
            meta.map_directory + "signal_unfiltered_freqs_nside128_" + str(id_sim).zfill(4) + ".npy"
        )
        noise_maps = np.load(
            "/pscratch/sd/b/beringue/BB-AWG/MEGATOP/1224_sims_obsmat_freqs/noise_nhits_freqs_nside128_"
            + str(meta.id_sim).zfill(4)
            + ".npy"
        )
        cmb_fg_freq_maps_beamed = beamed_sky_unfiltered + noise_maps
    elif hasattr(meta, "noiseless_filtered") and meta.noiseless_filtered:
        meta.logger.info("NOISELESS FILTERED CASE")
        try:
            cmb_fg_freq_maps_beamed = np.load(
                meta.map_directory + "total_freqs_nside128_" + str(id_sim).zfill(4) + ".npy"
            )
        except FileNotFoundError:
            cmb_fg_freq_maps_beamed = np.load(
                meta.map_directory + "total_beamed_freqs_nside128_" + str(id_sim).zfill(4) + ".npy"
            )

        noise_maps = np.load(
            "/pscratch/sd/b/beringue/BB-AWG/MEGATOP/1224_sims_obsmat_freqs/noise_nhits_freqs_nside128_"
            + str(meta.id_sim).zfill(4)
            + ".npy"
        )
        cmb_fg_freq_maps_beamed = cmb_fg_freq_maps_beamed - noise_maps
    else:
        try:
            cmb_fg_freq_maps_beamed = np.load(
                meta.map_directory + "total_beamed_freqs_nside128_" + str(id_sim).zfill(4) + ".npy"
            )
        except FileNotFoundError:
            cmb_fg_freq_maps_beamed = np.load(
                meta.map_directory + "total_freqs_nside128_" + str(id_sim).zfill(4) + ".npy"
            )

    return cmb_fg_freq_maps_beamed
