from pathlib import Path

import healpy as hp
import numpy as np
import numpy.typing as npt
import pymaster as nmt

from megatop.utils.compsep import truncate_alm

from .logger import logger
from .timer import function_timer


@function_timer("common-beam-and-nside")
def common_beam_and_nside(
    nside: int,
    common_beam: float,
    frequency_beams: list[float],
    freq_maps: list[npt.ArrayLike],
    output_alms: bool = False,
    lmax_convolution: int | None = None,
    DEBUGtruncatealms: bool = False,
    DEBUGlm_range: tuple[int, int] | None = None,
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
    freq_alms_out = []
    logger.info(f"Common beam correction -> {common_beam} arcmin and NSIDE -> {nside}")
    if lmax_convolution is None:
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

        if DEBUGtruncatealms:
            logger.warning("WARNING WARNING WARNING WARNING WARNING WARNING")
            logger.warning("DEBUG TRUNCATE ALMS IS ON")
            logger.warning("WARNING WARNING WARNING WARNING WARNING WARNING")
            if DEBUGlm_range is not None:
                lmin, lmax = DEBUGlm_range
                from megatop.utils.compsep import (
                    set_alm_tozero_above_lmax,
                    set_alm_tozero_below_lmin,
                )

                alm_out_T = set_alm_tozero_above_lmax(alm_out_T, lmax)
                alm_out_E = set_alm_tozero_above_lmax(alm_out_E, lmax)
                alm_out_B = set_alm_tozero_above_lmax(alm_out_B, lmax)

                alm_out_T = set_alm_tozero_below_lmin(alm_out_T, lmin)
                alm_out_E = set_alm_tozero_below_lmin(alm_out_E, lmin)
                alm_in_B = set_alm_tozero_below_lmin(alm_out_B, lmin)

        if output_alms:
            freq_alms_out.append([alm_out_T, alm_out_E, alm_out_B])

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
    if output_alms:
        return np.array(freq_maps_out), np.array(freq_alms_out)

    return np.array(freq_maps_out)


def read_input_maps(list_mapnames: list[Path]) -> list[npt.ArrayLike]:
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


@function_timer("alm_common_beam")
def alm_common_beam(
    nside: int,
    common_beam: float,
    frequency_beams: list[float],
    freq_maps: list[npt.ArrayLike],
    analysis_mask: npt.ArrayLike | None = None,
    harmonic_analysis_lmax: int | None = None,
):
    mean_fsky = np.mean(analysis_mask**2)  # the analysis mask must be normalized!
    mean_fsky_correction = np.sqrt(mean_fsky)

    data_alms = []

    for f in range(freq_maps.shape[0]):
        fields = nmt.NmtField(
            analysis_mask,
            freq_maps[f, 1:],
            beam=None,
            purify_e=False,
            purify_b=False,
            n_iter=10,
        )
        # The smooth mask (mask_analysis) is applied in the NmtField constructor
        data_alms.append(truncate_alm(fields.alm, lmax_new=harmonic_analysis_lmax - 1))
    data_alms = np.array(data_alms) / mean_fsky_correction

    common_beam_ell = hp.gauss_beam(
        np.radians(common_beam / 60.0),
        lmax=3 * nside,
        pol=True,
    )[
        :-1, 1
    ]  # taking only the GRAD/ELECTRIC/E polarization beam (it is equal to the  CURL/MAGNETIC/B polarization beam)

    beam4namaster = np.array(
        [
            hp.gauss_beam(np.radians(beam / 60), lmax=3 * nside, pol=True)[:-1, 1] / common_beam_ell
            for beam in frequency_beams
        ]
    )
    beam4namaster = beam4namaster[..., : harmonic_analysis_lmax - 1]

    assert beam4namaster.shape[-1] == hp.Alm.getlmax(data_alms.shape[-1]), (
        f"beam4namaster shape {beam4namaster.shape} does not match data_alms shape {data_alms.shape}"
    )

    for f in range(data_alms.shape[0]):
        data_alms[f, 0] = hp.almxfl(data_alms[f, 0], 1 / beam4namaster[f])
        data_alms[f, 1] = hp.almxfl(data_alms[f, 1], 1 / beam4namaster[f])
    return data_alms


# def TF_correction_on_alms(
#     alm: npt.NDArray,
#     TF: npt.NDArray,):
#     """
#     Apply TF correction on alms.
#     Parameters
#     ----------
#     alm : npt.NDArray
#         Alms to apply TF correction on.
#     TF : npt.NDArray
#         TF correction to apply on alms.
#     Returns
#     -------
#     npt.NDArray
#         Alms with TF correction applied.
#     """
