from pathlib import Path

import healpy as hp
import numpy as np

from .logger import logger
from .timer import function_timer


@function_timer("common-beam-and-nside")
def common_beam_and_nside(
    nside: int,
    common_beam: float,
    frequency_beams: list[float],
    freq_maps: list[np.typing.ArrayLike],
):
    nside_input_maps = [hp.npix2nside(freq_maps[i].shape[-1]) for i in range(len(frequency_beams))]
    idx_nside_small = np.argwhere(np.array(nside_input_maps) < nside)
    if idx_nside_small.size > 0:
        logger.error("Some input maps have too small nsides")
        logger.error("Check your yaml !")
        logger.error("Exiting")
        msg = "Some of input maps have too small nside."
        raise ValueError(msg)  # TODO better error handling ?

    freq_maps_out = []
    logger.info(f"Common beam correction -> {common_beam} arcmin and NSIDE -> {nside}")
    lmax_convolution = 3 * nside
    wpix_out = hp.pixwin(
        nside, pol=True, lmax=lmax_convolution
    )  # Pixel window function of output maps
    Bl_gauss_common = hp.gauss_beam(
        np.radians(common_beam / 60.0),
        lmax=lmax_convolution,
        pol=True,
    )

    for i_beam, beam in enumerate(frequency_beams):
        # Input window function. WARNING: Must be done in the loop, as input map don't necessarily have the same nside (e.g. MSS2)
        logger.info(f"Input beam FWHM {beam} arcmin -> {common_beam} arcmin")
        wpix_in = hp.pixwin(
            hp.get_nside(freq_maps[i_beam][0]), pol=True, lmax=lmax_convolution
        )  # Pixel window function of input maps
        wpix_in[1][0:2] = 1.0  # in order not to divide by 0

        # beam corrections
        Bl_gauss_fwhm = hp.gauss_beam(np.radians(beam / 60), lmax=lmax_convolution, pol=True)

        bl_correction = Bl_gauss_common / Bl_gauss_fwhm

        sm_corr_T = bl_correction[:, 0] * wpix_out[0] / wpix_in[0]
        sm_corr_P = bl_correction[:, 1] * wpix_out[1] / wpix_in[1]

        # map-->alm
        map_in_T = np.array(freq_maps[i_beam][0])
        map_in_Q = np.array(freq_maps[i_beam][1])
        map_in_U = np.array(freq_maps[i_beam][2])

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
            nside,
            lmax=lmax_convolution,
            pixwin=False,
            fwhm=0.0,
            pol=True,
        )

        # a priori all the options are set to there default, even lmax which is computed wrt input alms
        out_map = np.array([map_out_T, map_out_Q, map_out_U])
        freq_maps_out.append(out_map)
    return np.array(freq_maps_out)


def read_input_maps(list_mapnames: list[Path]) -> list[np.typing.ArrayLike]:
    """
    This function reads the frequency maps from the files and returns them as an array.

    Parameters
    ----------
    list_mapnames: list
        list of paths to maps

    Returns
    -------
        freq_maps_input : (list)
            The list of frequency maps, with len (num_freq), each of them having shape (num_stokes, num_pixels).s
    Notes
    -----
    * The input maps are allowed to have different `num_pixels' to allow for different resolutions.
    """
    freq_maps_input = []
    for mapname in list_mapnames:
        logger.debug(f"Reading map from {mapname}")
        freq_maps_input.append(hp.read_map(mapname, field=None))
    return freq_maps_input
