import os
from pathlib import Path

import healpy as hp
import numpy as np
import numpy.typing as npt
import pymaster as nmt

from megatop.utils.compsep import set_alm_tozero_above_lmax, set_alm_tozero_below_lmin

from .logger import logger
from .timer import function_timer

HEALPY_DATA_PATH = os.getenv("HEALPY_LOCAL_DATA", None)


def _apply_npipe_unbeaming(
    alms: tuple, frequency: int | str, npipe_beam_path: Path, lmax: int | None = None
):
    """Attempt to reverse NPIPE beam + T->P leakage from a set of alms.

    alms: (alm_T, alm_E, alm_B)
    frequency: frequency tag (int or str) used to find the proper beam/leak files
    npipe_beam_path: directory containing beam/leakage files
    """
    import glob

    alm_T, alm_E, alm_B = alms

    freq_str = f"{int(frequency):03d}" if isinstance(frequency, int) or str(frequency).isdigit() else str(frequency)

    p = Path(npipe_beam_path)
    if not p.exists():
        raise FileNotFoundError(f"NPIPE beam path {p} does not exist")

    # Construct expected filenames (NPIPE naming convention provided by user)
    # e.g. Bl_TEB_npipe6v20_030GHzx030GHz.fits and Wl_npipe6v20_030GHzx030GHz.fits
    bl_name = f"Bl_TEB_npipe6v20_{freq_str}GHzx{freq_str}GHz.fits"
    wl_name = f"Wl_npipe6v20_{freq_str}GHzx{freq_str}GHz.fits"

    bl_file = p / bl_name
    wl_file = p / wl_name

    if not bl_file.exists():
        raise FileNotFoundError(f"Could not find expected NPIPE beam file '{bl_name}' in {p}")

    # load arrays: try healpy.read_cl then numpy.load
    def _load_cl(path: Path):
        try:
            return hp.read_cl(str(path))
        except Exception:
            try:
                data = np.load(path, allow_pickle=True)
                # if npz-like, try to extract arrays
                if isinstance(data, np.lib.npyio.NpzFile):
                    # pick first array
                    return data[list(data.keys())[0]]
                return data
            except Exception:
                # try text
                return np.loadtxt(path)

    bl = hp.read_cl(str(bl_file))
    wl = None
    if wl_file is not None:
        wl = hp.read_cl(str(wl_file))

    # perform debeaming and T->P leakage subtraction using the same approach as the user-provided snippet
    alm_corr_T = alm_T.copy()
    alm_corr_E = alm_E.copy()
    alm_corr_B = alm_B.copy()

    # inverse beam for T
    inv_bl_T = 1.0 / bl[0]
    T_debeamed = hp.almxfl(alm_corr_T.copy(), inv_bl_T)

    if wl is not None:
        # apply leakage correction to pol components
        for i in range(1, 3):
            w = wl[i].ravel().copy()
            w[w < 0] = 0
            w = np.sqrt(w)
            leak = hp.almxfl(T_debeamed, w)
            if i == 1:
                alm_corr_E -= leak
            else:
                alm_corr_B -= leak

    # debeam all components
    alm_debeam_T = alm_corr_T.copy()
    alm_debeam_E = alm_corr_E.copy()
    alm_debeam_B = alm_corr_B.copy()

    for i, alm in enumerate((alm_debeam_T, alm_debeam_E, alm_debeam_B)):
        inv_bl = 1.0 / bl[i]
        hp.almxfl(alm, inv_bl, inplace=True)

    return alm_debeam_T, alm_debeam_E, alm_debeam_B



@function_timer("common-beam-and-nside")
def common_beam_and_nside(
    nside: int,
    common_beam: float,
    frequency_beams: list[float],
    freq_maps: list[npt.ArrayLike],
    lmax: int,
    frequency_tags: list[int] | None = None,
    output_alms: bool = False,
    DEBUGtruncatealms: bool = False,
    DEBUGlm_range: tuple[int, int] | None = None,
    npipe_beam_correction: bool = False,
    npipe_beam_path: Path | None = None,
):
    # TODO: remove DEBUGtruncatealms and DEBUGlm_range after testing
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
            hp.get_nside(freq_maps[i_beam][0]),
            pol=True,
            lmax=lmax,
            # datapath=HEALPY_DATA_PATH,
        )  # Pixel window function of input maps
        wpix_in[1][0:2] = 1.0  # in order not to divide by 0

        # beam corrections
        Bl_gauss_fwhm = hp.gauss_beam(np.radians(beam / 60), lmax=lmax, pol=True)

        # If we applied the NPIPE unbeamning earlier, the input alms/maps are effectively
        # deconvolved of the original input beam. In that case we should NOT divide by
        # the input (frequency) beam here — we only want to apply the common beam.
        if npipe_beam_correction:
            bl_correction = Bl_gauss_common
        else:
            bl_correction = Bl_gauss_common / Bl_gauss_fwhm

        sm_corr_T = bl_correction[:, 0] * wpix_out[0] / wpix_in[0]
        sm_corr_P = bl_correction[:, 1] * wpix_out[1] / wpix_in[1]

        # map-->alm
        map_in_T = np.array(freq_maps[i_beam][0])
        map_in_Q = np.array(freq_maps[i_beam][1])
        map_in_U = np.array(freq_maps[i_beam][2])

        alm_in_T, alm_in_E, alm_in_B = hp.map2alm(
            [map_in_T, map_in_Q, map_in_U],
            lmax=lmax,
            pol=True,
            iter=10,
            datapath=HEALPY_DATA_PATH,
        )
        #   apply NPIPE-specific reverse beam + T->P leakage correction
        if npipe_beam_correction and npipe_beam_path is not None and frequency_tags is not None:
            try:
                alm_in_T, alm_in_E, alm_in_B = _apply_npipe_unbeaming(
                    (alm_in_T, alm_in_E, alm_in_B),
                    frequency=frequency_tags[i_beam],
                    npipe_beam_path=Path(npipe_beam_path),
                    lmax=lmax_convolution,
                )
                logger.info(f"Applied NPIPE unbeam/leakage correction for freq index {i_beam}")
            except Exception as exc:  # pragma: no cover - best-effort; don't break pipeline
                logger.warning(f"NPIPE unbeam correction failed: {exc}; continuing without it")
        elif npipe_beam_correction:
            logger.warning(
                "NPIPE beam correction requested but frequency_tags or npipe_beam_path is missing; skipping NPIPE correction"
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
            lmax=lmax,
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
    common_beam: float,
    frequency_beams: list[float],
    freq_maps: list[npt.ArrayLike],
    analysis_mask: npt.ArrayLike,
    harmonic_analysis_lmax: int,
    frequency_tags: list[int] | None = None,
    npipe_beam_correction: bool = False,
    npipe_beam_path: Path | None = None,
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

    if npipe_beam_correction:
        # Inputs were debeamed already; beam4namaster should be 1/common_beam_ell so that
        # applying 1/beam4namaster multiplies by common_beam_ell.
        beam4namaster = np.array([1.0 / common_beam_ell for _ in frequency_beams])
    else:
        beam4namaster = np.array(
            [
                hp.gauss_beam(np.radians(beam / 60), lmax=2 * nside, pol=True)[:-1, 1] / common_beam_ell
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
