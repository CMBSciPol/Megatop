import os
from pathlib import Path

import healpy as hp
import numpy as np
import numpy.typing as npt
import pymaster as nmt

import megatop.utils.harmonic as hu
from megatop.utils.compsep import set_alm_tozero_above_lmax, set_alm_tozero_below_lmin

from .logger import logger
from .timer import function_timer

HEALPY_DATA_PATH = os.getenv("HEALPY_LOCAL_DATA", None)


@function_timer("common-beam-and-nside")
def common_beam_and_nside(
    nside: int,
    common_beam: float,
    frequency_beams: list[float],
    freq_maps: list[npt.ArrayLike],
    lmax: int,
    output_alms: bool = False,
    DEBUGtruncatealms: bool = False,
    DEBUGlm_range: tuple[int, int] | None = None,
):
    # TODO: remove DEBUGtruncatealms and DEBUGlm_range after testing
    nside_input_maps = [hp.npix2nside(m.shape[-1]) for m in freq_maps]
    if any(n < nside for n in nside_input_maps):
        raise ValueError("Some input maps have smaller nside than target. Check your yaml.")

    freq_maps_out = []
    freq_alms_out = []
    logger.info(f"Common beam correction -> {common_beam} arcmin and NSIDE -> {nside}")

    wpix_out = hp.pixwin(
        nside,
        pol=True,
        lmax=lmax,
    )  # Pixel window function of output maps
    Bl_gauss_common = hp.gauss_beam(
        np.radians(common_beam / 60.0),
        lmax=lmax,
        pol=True,
    )

    for i_beam, beam in enumerate(frequency_beams):
        # Input window function. WARNING: Must be done in the loop, as input map don't necessarily have the same nside (e.g. MSS2)
        logger.info(f"Input beam FWHM {beam} arcmin -> {common_beam} arcmin")
        wpix_in = hp.pixwin(
            nside_input_maps[i_beam],
            pol=True,
            lmax=lmax,
            datapath=HEALPY_DATA_PATH,
        )  # Pixel window function of input maps
        wpix_in[1][0:2] = 1.0  # in order not to divide by 0

        # beam corrections
        Bl_gauss_fwhm = hp.gauss_beam(np.radians(beam / 60), lmax=lmax, pol=True)

        bl_correction = Bl_gauss_common / Bl_gauss_fwhm

        sm_corr_T = bl_correction[:, 0] * wpix_out[0] / wpix_in[0]
        sm_corr_P = bl_correction[:, 1] * wpix_out[1] / wpix_in[1]

        # map-->alm
        alms_in = hu.map2alm(freq_maps[i_beam], spin=[0, 2], lmax=lmax, niter=10)

        # change beam and wpix
        hu.almxfl(alms_in[0], sm_corr_T, inplace=True)
        hu.almxfl(alms_in[1:], sm_corr_P, inplace=True)

        if DEBUGtruncatealms:
            logger.warning("WARNING WARNING WARNING WARNING WARNING WARNING")
            logger.warning("DEBUG TRUNCATE ALMS IS ON")
            logger.warning("WARNING WARNING WARNING WARNING WARNING WARNING")
            if DEBUGlm_range is not None:
                lmin, lmax = DEBUGlm_range

                alms_in[0] = set_alm_tozero_above_lmax(alms_in[0], lmax)
                alms_in[1] = set_alm_tozero_above_lmax(alms_in[1], lmax)
                alms_in[2] = set_alm_tozero_above_lmax(alms_in[2], lmax)

                alms_in[0] = set_alm_tozero_below_lmin(alms_in[0], lmin)
                alms_in[1] = set_alm_tozero_below_lmin(alms_in[1], lmin)
                alms_in[2] = set_alm_tozero_below_lmin(alms_in[2], lmin)

        if output_alms:
            freq_alms_out.append(alms_in)

        # alm-->map
        out_map = hu.alm2map(alms_in, spin=[0, 2], nside=nside, lmax=lmax)

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
    common_beam: float,
    frequency_beams: list[float],
    freq_maps: list[npt.ArrayLike],
    analysis_mask: npt.ArrayLike,
    harmonic_analysis_lmax: int,
):
    data_alms = np.array(
        [
            nmt.NmtField(
                analysis_mask,
                freq_maps[f, 1:],
                beam=None,
                purify_e=False,
                purify_b=False,
                n_iter=10,
                lmax=harmonic_analysis_lmax,
                lmax_mask=harmonic_analysis_lmax,
            ).get_alms()
            for f in range(freq_maps.shape[0])
        ]
    )
    mean_fsky = np.mean(np.square(analysis_mask))  # the analysis mask must be normalized!
    data_alms /= np.sqrt(mean_fsky)

    common_beam_ell = hp.gauss_beam(
        np.radians(common_beam / 60.0),
        lmax=harmonic_analysis_lmax,
        pol=True,
    )[:, 1]  # E polarization beam; shape (harmonic_analysis_lmax + 1,)

    beam4namaster = np.array(
        [
            hp.gauss_beam(np.radians(beam / 60), lmax=harmonic_analysis_lmax, pol=True)[:, 1]
            / common_beam_ell
            for beam in frequency_beams
        ]
    )

    for f in range(data_alms.shape[0]):
        hu.almxfl(data_alms[f, :2], 1 / beam4namaster[f], inplace=True)
    return data_alms
