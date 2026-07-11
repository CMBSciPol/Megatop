import os
from typing import Literal

import camb
import healpy as hp
import numpy as np
import pymaster as nmt
from numpy.typing import NDArray
from pixell import enmap

from megatop.config import CAMBCosmoPars
from megatop.landscapes import AbstractLandscape
from megatop.utils import logger

HEALPY_DATA_PATH = os.getenv("HEALPY_LOCAL_DATA", None)


def _car_deconvolve_pixwin(maps: NDArray, wcs) -> enmap.ndmap:
    """Deconvolve the CAR pixel window from `maps` in map space.

    The mocker applies the (anisotropic) CAR pixel window with `enmap.apply_window`;
    NaMaster's `beam=` only takes a 1-D `b_ell`, so the window cannot be folded there.
    We remove it symmetrically (`pow=-1`) before building the `NmtField`, leaving only
    the Gaussian beam to be deconvolved by NaMaster.
    """
    return enmap.apply_window(enmap.enmap(np.asarray(maps), wcs), pow=-1)


CAMBSpectraKey = Literal[
    "total",
    "unlensed_scalar",
    "unlensed_total",
    "lensed_scalar",
    "tensor",
    "lens_potential",
]


def compute_spectra_from_camb(
    r: float,
    cosmo_pars: CAMBCosmoPars,
    which: CAMBSpectraKey = "total",
):
    """
    Compute CMB TT, EE, BB, TE spectra from CAMB.
    Parameters
    ----------
    r : float
        Tensor-to-scalar ratio.
    cosmo_pars : CAMBCosmoPars
        Cosmological parameters passed to camb.set_params().
    which : str
        One of the spectra keys returned by results.get_cmb_power_spectra().
    Returns
    -------
    TT, EE, BB, TE : array-like
        Each with shape (LMAX+1,)
    """

    cosmo_params = camb.set_params(**cosmo_pars.as_camb_kwargs())
    cosmo_params.set_for_lmax(2000, lens_potential_accuracy=1)
    cosmo_params.WantTensors = True

    infl_params = camb.initialpower.InitialPowerLaw()
    infl_params.set_params(
        ns=cosmo_pars.ns,
        r=r,
        parameterization="tensor_param_indeptilt",
        nt=0,
        ntrun=0,
    )
    cosmo_params.InitPower = infl_params

    results = camb.get_results(cosmo_params)
    full_spectra = results.get_cmb_power_spectra(cosmo_params, CMB_unit="muK", raw_cl=True)

    cmb = full_spectra[which]

    TT = cmb[:, 0]
    EE = cmb[:, 1]
    BB = cmb[:, 2]
    TE = cmb[:, 3]

    return TT, EE, BB, TE


def compute_auto_cross_cl_from_maps_dict(
    maps_dict: dict,
    analysis_mask: NDArray,
    workspace,
    beam: NDArray,
    n_iter: int,
    lmax: int,
    purify_b: bool,
    purify_e: bool,
    inverse_effective_transfer_function: NDArray | None = None,
    *,
    landscape: AbstractLandscape,
):
    # A CAR landscape selects CAR fields (pymaster builds them on the CAR geometry from
    # its wcs); the CAR pixel window is removed in map space beforehand.
    fields = []
    for key in maps_dict:
        maps = (
            _car_deconvolve_pixwin(maps_dict[key], landscape.wcs)
            if landscape.is_car
            else maps_dict[key]
        )
        fields.append(
            nmt.NmtField(
                mask=analysis_mask,
                maps=maps,
                spin=2,
                beam=beam,
                purify_e=purify_e,
                purify_b=purify_b,
                n_iter=n_iter,
                n_iter_mask=n_iter,
                lmax=lmax,
                lmax_mask=lmax,
                wcs=landscape.wcs,
            )
        )

    # Compute the power spectra
    cl_list = []
    cl_matrix = None
    for i, f_a in enumerate(fields):
        for j, f_b in enumerate(fields):
            if i <= j:
                cl_coupled = nmt.compute_coupled_cell(f_a, f_b)
                cl_decoupled = workspace.decouple_cell(cl_coupled)
                cl_list.append(cl_decoupled)
                if cl_matrix is None:
                    cl_matrix = np.zeros(
                        (len(fields), len(fields), cl_decoupled.shape[0], cl_decoupled.shape[-1])
                    )
                if inverse_effective_transfer_function is not None:
                    cl_matrix[i, j] = cl_decoupled
                    if i != j:
                        cl_matrix[j, i] = cl_decoupled

    cl_dict = {}
    if inverse_effective_transfer_function is None:
        # Store in dictionary with key x key
        for i, key1 in enumerate(maps_dict.keys()):
            for j, key2 in enumerate(maps_dict.keys()):
                if i <= j:
                    cl_dict[key1 + "x" + key2] = cl_list.pop(0)

    else:
        logger.info("Applying effective transfer function to the spectra")
        tf_corrected_cl_matrix = np.einsum(
            "ckijl,ckjl->ckil", inverse_effective_transfer_function, cl_matrix
        )
        for i, key1 in enumerate(maps_dict.keys()):
            for j, key2 in enumerate(maps_dict.keys()):
                if i <= j:
                    cl_dict[key1 + "x" + key2] = tf_corrected_cl_matrix[i, j]

    return cl_dict


def get_common_beam_wpix(common_beam_fwhm_arcmin: float, landscape, lmax: int):
    Bl_gauss_common = hp.gauss_beam(np.radians(common_beam_fwhm_arcmin / 60.0), lmax=lmax, pol=True)
    if landscape.is_car:
        # CAR pixel window is anisotropic and handled in map space; the NaMaster
        # beam is the Gaussian common beam only.
        return Bl_gauss_common[:, 1]
    wpix_out = hp.pixwin(
        landscape.working_nside(lmax), pol=True, lmax=lmax, datapath=HEALPY_DATA_PATH
    )  # Pixel window function of output maps
    return Bl_gauss_common[:, 1] * wpix_out[1]  # TODO only polarisation one ?


def initialize_nmt_workspace(
    nmt_bins,
    analysis_mask: NDArray,
    beam: NDArray,
    purify_e: bool,
    purify_b: bool,
    n_iter: int,
    lmax: int,
    *,
    landscape: AbstractLandscape,
):
    fields_init_wsp = nmt.NmtField(
        mask=analysis_mask,
        maps=None,
        spin=2,
        beam=beam,
        purify_e=purify_e,
        purify_b=purify_b,
        n_iter=n_iter,
        n_iter_mask=n_iter,
        lmax=lmax,
        lmax_mask=lmax,
        wcs=landscape.wcs,
    )
    return nmt.NmtWorkspace.from_fields(fields_init_wsp, fields_init_wsp, nmt_bins)


def limit_namaster_output(all_Cls, bin_index_lminlmax):
    """
    This function limits the output of namaster to the desired l range.

    Args:
        all_Cls (dict): The dictionary containing the Cls computed by namaster.
        bin_index_lminlmax (ndarray): The indices of the bins corresponding to the desired l range.

    Returns:
        dict: The dictionary containing the Cls computed by namaster, limited to the desired l range.
    """
    all_Cls_limited = {}
    for key, value in all_Cls.items():
        all_Cls_limited[key] = value[..., bin_index_lminlmax]
    return all_Cls_limited


def spectra_from_namaster(
    freq_noise_maps,
    mask_analysis,
    workspaceff,
    nmt_bins,
    compute_cross_freq=False,
    purify_e=False,
    purify_b=False,
    beam=None,
    return_all_spectra=False,
    lmax=None,
    *,
    landscape: AbstractLandscape,
):
    """
    Computes the auto and cross-spectra from the frequency noise maps using NaMaster.
    Parameters
    ----------
    freq_noise_maps : np.ndarray
        Frequency noise maps, shape (n_freq, 3, n_pix).
        where 3 refers to the T, Q, U components.
    mask_analysis : np.ndarray
        Analysis mask, shape (n_pix,).
    workspaceff : nmt.NmtWorkspace
        NaMaster workspace for decoupling the spectra.
    nmt_bins : nmt.NmtBin
        NaMaster binning object for the spectra.
    compute_cross_freq : bool, optional
        Whether to compute cross-frequency spectra. Default is False.
    purify_e : bool, optional
        Whether to purify E-mode polarization. Default is False.
    purify_b : bool, optional
        Whether to purify B-mode polarization. Default is False.
    beam : np.ndarray, optional
        Beam correction factors, shape (n_freq, n_bins). If None, no beam correction is applied.
        Default is None.
    Returns
    -------
    cl_decoupled_freq : np.ndarray
        Decoupled power spectra for each frequency, shape (n_freq, n_bins, 3).
        Where 3 refers to the T, E, B components. And T is set to zero.
    unbin_cl_decoupled_freq : np.ndarray
        Unbinned decoupled power spectra for each frequency, shape (n_freq, n_bins, 3).
        Where 3 refers to the T, E, B components. And T is set to zero.

    Notes
    -----
    - The function computes the auto-spectra for each frequency noise map.
    - The T component is set to zero in the output spectra.
    - The EB cross-spectra are ignored in the output.
    """

    if compute_cross_freq:
        msg = "Cross-frequency spectra computation is not implemented yet"
        raise NotImplementedError(msg)

    if beam is not None and beam.shape[0] != freq_noise_maps.shape[0]:
        msg = f"Beam shape {beam.shape} does not match frequency noise maps shape {freq_noise_maps.shape}"
        raise ValueError(msg)

    # reset_workspace = True if workspaceff is None else False
    reset_workspace = (
        workspaceff is None
    )  # returns bool depending on whether workspaceff is None or not

    cl_decoupled_freq = []
    unbin_cl_decoupled_freq = []
    for f in range(freq_noise_maps.shape[0]):
        beam_f = beam[f] if beam is not None else None

        maps_f = freq_noise_maps[f, 1:]
        if landscape.is_car:
            maps_f = _car_deconvolve_pixwin(maps_f, landscape.wcs)
        fields = nmt.NmtField(
            mask_analysis,
            maps_f,
            beam=beam_f,
            purify_e=purify_e,
            purify_b=purify_b,
            n_iter=10,
            lmax=lmax,
            lmax_mask=lmax,
            wcs=landscape.wcs,
        )
        if reset_workspace:
            workspaceff = nmt.NmtWorkspace.from_fields(fields, fields, nmt_bins)

        cl_coupled = nmt.compute_coupled_cell(fields, fields)
        cl_decoupled = workspaceff.decouple_cell(cl_coupled)
        unbin_cl_decoupled = nmt_bins.unbin_cell(cl_decoupled)

        if return_all_spectra:
            # Append the full decoupled and unbinned spectra
            cl_decoupled_freq.append(cl_decoupled)
            unbin_cl_decoupled_freq.append(unbin_cl_decoupled)
        else:
            # Keeping only the TT, EE, BB components, setting T to zero
            # Warning: we are ignoring the EB cross-spectra here
            cl_decoupled_freq.append([cl_decoupled[0] * 0, cl_decoupled[0], cl_decoupled[3]])
            unbin_cl_decoupled_freq.append(
                [unbin_cl_decoupled[0] * 0, unbin_cl_decoupled[0], unbin_cl_decoupled[3]]
            )

    return np.array(cl_decoupled_freq), np.array(unbin_cl_decoupled_freq)


def get_effective_transfer_function(
    transfer_freq: NDArray, W_maxL: NDArray, binary_mask: NDArray | None = None
) -> NDArray:
    """
    Computes the effective transfer function from the transfer functions at each frequency and
    the maximum likelihood results for the component separation operator
    Parameters
    ----------
    transfer_freq : np.ndarray
        Frequency transfer functions, shape (n_freq, 9 , 9, n_bins). Where 9 refers to the T, E, B, and their cross-spectra (xy->wz).
    W_maxL : np.ndarray
        Component separation operator, shape (ncomp, nfreq, n_stokes, n_pix).
    binary_mask : np.ndarray, optional
        Binary mask, shape (n_pix,). If provided, the effective transfer function is computed
        only over the observed pixels. Default is None, which means the effective transfer function is computed
        over all pixels.
    Returns
    -------
    effective_transfer_function : np.ndarray
        Effective transfer function, shape (ncomp, ncomp, pol_spectra, pol_spectra ,nbin). pol_spectra refers EE, EB, BE, BB
    """

    #  Averaging W over observed pixels and stokes parameters:
    if binary_mask is None:
        pix_stokes_mean_W = np.mean(W_maxL, axis=(-2, -1))  # shape (ncomp, nfreq)
    else:
        pix_stokes_mean_W = np.mean(W_maxL[..., binary_mask], axis=(-2, -1))  # shape (ncomp, nfreq)

    pol_transfer = transfer_freq[:, -4:, -4:]  # Keeping only polarised components (EE, EB, BE, BB)
    # applying comp-sep operator on both sides of the transfer function
    normalisation_factor = np.einsum(
        "cf, fk -> ck", pix_stokes_mean_W, pix_stokes_mean_W.T
    )  # shape (ncomp, pol_spectra, n_bins)

    effective_transfer_function = np.einsum(
        "cf,fsdl,fk->cksdl", pix_stokes_mean_W, pol_transfer, pix_stokes_mean_W.T
    )  # shape (ncomp, pol_spectra, n_bins)

    # applying normalisation_factor to each comp x comp in effective_transfer_function
    # to get the effective transfer function
    normalized_effective_transfer_function = (
        effective_transfer_function
        / normalisation_factor[(...,) + (np.newaxis,) * (effective_transfer_function.ndim - 2)]
    )

    # inverting over polarised spectra space:
    inverse_effective_transfer_function = np.zeros_like(normalized_effective_transfer_function)
    for i in range(normalized_effective_transfer_function.shape[0]):
        for j in range(normalized_effective_transfer_function.shape[1]):
            for ell in range(normalized_effective_transfer_function.shape[-1]):
                # Inverting the transfer function for each ell over the spectra dimension
                inverse_effective_transfer_function[i, j, :, :, ell] = np.linalg.inv(
                    normalized_effective_transfer_function[i, j, :, :, ell]
                )

    return normalized_effective_transfer_function, inverse_effective_transfer_function


def get_effective_transfer_function_WCl(
    transfer_freq: NDArray, W_maxL: NDArray, binary_mask: NDArray | None = None
) -> NDArray:
    """
    Computes the effective transfer function from the transfer functions at each frequency and
    the maximum likelihood results for the component separation operator
    Parameters
    ----------
    transfer_freq : np.ndarray
        Frequency transfer functions, shape (n_freq, 9 , 9, n_bins). Where 9 refers to the T, E, B, and their cross-spectra (xy->wz).
    W_maxL : np.ndarray
        Component separation operator, shape (ncomp, nfreq, n_stokes, n_pix).
    binary_mask : np.ndarray, optional
        Binary mask, shape (n_pix,). If provided, the effective transfer function is computed
        only over the observed pixels. Default is None, which means the effective transfer function is computed
        over all pixels.
    Returns
    -------
    effective_transfer_function : np.ndarray
        Effective transfer function, shape (ncomp, ncomp, pol_spectra, pol_spectra ,nbin). pol_spectra refers EE, EB, BE, BB
    """

    #  Averaging W over observed pixels and stokes parameters:
    if binary_mask is None:
        pix_stokes_mean_W = np.mean(W_maxL, axis=(-2, -1))  # shape (ncomp, nfreq)
    else:
        pix_stokes_mean_W = np.mean(W_maxL[..., binary_mask], axis=(-2, -1))  # shape (ncomp, nfreq)

    pol_transfer = transfer_freq[:, -4:, -4:]  # Keeping only polarised components (EE, EB, BE, BB)
    # applying comp-sep operator on both sides of the transfer function
    normalisation_factor = np.einsum(
        "cf, fk -> ck", pix_stokes_mean_W, pix_stokes_mean_W.T
    )  # shape (ncomp, pol_spectra, n_bins)

    effective_transfer_function = np.einsum(
        "cf,fsdl,fk->cksdl", pix_stokes_mean_W, pol_transfer, pix_stokes_mean_W.T
    )  # shape (ncomp, pol_spectra, n_bins)

    # applying normalisation_factor to each comp x comp in effective_transfer_function
    # to get the effective transfer function
    normalized_effective_transfer_function = (
        effective_transfer_function
        / normalisation_factor[(...,) + (np.newaxis,) * (effective_transfer_function.ndim - 2)]
    )

    # inverting over polarised spectra space:
    inverse_effective_transfer_function = np.zeros_like(normalized_effective_transfer_function)
    for i in range(normalized_effective_transfer_function.shape[0]):
        for j in range(normalized_effective_transfer_function.shape[1]):
            for ell in range(normalized_effective_transfer_function.shape[-1]):
                # Inverting the transfer function for each ell over the spectra dimension
                inverse_effective_transfer_function[i, j, :, :, ell] = np.linalg.inv(
                    normalized_effective_transfer_function[i, j, :, :, ell]
                )

    return normalized_effective_transfer_function, inverse_effective_transfer_function
