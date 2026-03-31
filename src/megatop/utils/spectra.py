import camb
import healpy as hp
import numpy as np
import pymaster as nmt
from numpy.typing import NDArray

from megatop import Config
from megatop.utils import logger


def compute_spectra_from_camb(r, cosmo_params_dict, which="total"):
    """
    Compute CMB TT, EE, BB, TE spectra from CAMB.
    Parameters
    ----------
    r : float
        Tensor-to-scalar ratio.
    cosmo_params_dict : dict
        Dictionary of cosmological parameters passed to camb.set_params().
    which : str
        One of the spectra keys returned by results.get_cmb_power_spectra().
    Returns
    -------
    TT, EE, BB, TE : array-like
        Each with shape (LMAX+1,)
    """

    cosmo_params = camb.set_params(**cosmo_params_dict)

    cosmo_params.set_for_lmax(2000, lens_potential_accuracy=1)
    cosmo_params.WantTensors = True

    infl_params = camb.initialpower.InitialPowerLaw()
    infl_params.set_params(
        ns=cosmo_params_dict["ns"],
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


def compute_auto_cross_cl_from_maps_list(
    maps_dict,
    mask,
    beam,
    workspace,
    purify_e=True,
    purify_b=True,
    n_iter=3,
    inverse_effective_transfer_function=None,
):
    # Create the fields
    fields = []
    for key in maps_dict:
        fields.append(
            nmt.NmtField(
                mask, maps_dict[key], beam=beam, purify_e=purify_e, purify_b=purify_b, n_iter=n_iter
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
        for i, key in enumerate(maps_dict.keys()):
            for j, key2 in enumerate(maps_dict.keys()):
                if i <= j:
                    cl_dict[key + "x" + key2] = cl_list.pop(0)

    else:
        logger.info("Applying effective transfer function to the spectra")
        tf_corrected_cl_matrix = np.einsum(
            "ckijl,ckjl->ckil", inverse_effective_transfer_function, cl_matrix
        )
        for i, key in enumerate(maps_dict.keys()):
            for j, key2 in enumerate(maps_dict.keys()):
                if i <= j:
                    cl_dict[key + "x" + key2] = tf_corrected_cl_matrix[i, j]

    return cl_dict


def compute_auto_cross_cl_from_alms_list(
    alms_dict,
    mask,
    beam,
    workspace,
    purify_e=True,
    purify_b=True,
    n_iter=3,
    # inverse_effective_transfer_function=None,
):
    # Create the fields
    # import time
    # import IPython; IPython.embed()
    fields = []
    for key in alms_dict:
        fields.append(
            nmt.NmtField(
                mask,
                [np.zeros_like(mask), np.zeros_like(mask)],
                beam=beam,
                purify_e=purify_e,
                purify_b=purify_b,
                n_iter=n_iter,
            )
        )
        lmax_field = hp.Alm.getlmax(fields[-1].alm.shape[-1])
        lmax_input_alm = hp.Alm.getlmax(alms_dict[key].shape[-1])
        # lmax field is by default 3*nside -1 while alm out of compsep is lmax = 2*nside (or whatever harmonic_lmax is set to)
        # we need to pad the alm to match the lmax of the field

        # start = time.time()
        if lmax_field > lmax_input_alm:
            for ell in range(lmax_input_alm):
                for m in range(ell + 1):
                    fields[-1].alm[0, hp.Alm.getidx(lmax_field, ell, m)] = alms_dict[key][
                        0, hp.Alm.getidx(lmax_input_alm, ell, m)
                    ]
                    fields[-1].alm[1, hp.Alm.getidx(lmax_field, ell, m)] = alms_dict[key][
                        1, hp.Alm.getidx(lmax_input_alm, ell, m)
                    ]
        # print("Time to pad alms: ", time.time() - start)

        # fields[-1].alm = alms_dict[key]

    # Compute the power spectra
    cl_list = []
    # cl_matrix = None
    for i, f_a in enumerate(fields):
        for j, f_b in enumerate(fields):
            if i <= j:
                cl_coupled = nmt.compute_coupled_cell(f_a, f_b)
                cl_decoupled = workspace.decouple_cell(cl_coupled)
                cl_list.append(cl_decoupled)
                # if cl_matrix is None:
                #     cl_matrix = np.zeros(
                #         (len(fields), len(fields), cl_decoupled.shape[0], cl_decoupled.shape[-1])
                #     )
                # # if inverse_effective_transfer_function is not None:
                #     cl_matrix[i, j] = cl_decoupled
                #     if i != j:
                #         cl_matrix[j, i] = cl_decoupled

    cl_dict = {}
    # if inverse_effective_transfer_function is None:
    # Store in dictionary with key x key
    for i, key in enumerate(alms_dict.keys()):
        for j, key2 in enumerate(alms_dict.keys()):
            if i <= j:
                cl_dict[key + "x" + key2] = cl_list.pop(0)

    # else:
    #     logger.info("Applying effective transfer function to the spectra")
    #     tf_corrected_cl_matrix = np.einsum(
    #         "ckijl,ckjl->ckil", inverse_effective_transfer_function, cl_matrix
    #     )
    #     for i, key in enumerate(alms_dict.keys()):
    #         for j, key2 in enumerate(alms_dict.keys()):
    #             if i <= j:
    #                 cl_dict[key + "x" + key2] = tf_corrected_cl_matrix[i, j]

    return cl_dict


def get_common_beam_wpix(common_beam_fwhm_arcmin, nside):
    wpix_out = hp.pixwin(nside, pol=True, lmax=3 * nside)  # Pixel window function of output maps
    Bl_gauss_common = hp.gauss_beam(
        np.radians(common_beam_fwhm_arcmin / 60), lmax=3 * nside, pol=True
    )

    return Bl_gauss_common[:, 1] * wpix_out[1]  # TODO only polarisation one ?


def get_effective_beam_noise_preproc(config: Config, A):
    lmax_convolution = 3 * config.nside
    wpix_out = hp.pixwin(
        config.nside, pol=True, lmax=lmax_convolution
    )  # Pixel window function of output maps
    Bl_gauss_common = 1

    beam_correction = []
    for i_f in range(len(config.frequencies)):
        Bl_gauss_fwhm = hp.gauss_beam(
            np.radians(config.beams[i_f] / 60), lmax=lmax_convolution, pol=True
        )
        bl_correction = Bl_gauss_common / Bl_gauss_fwhm

        sm_corr_P = bl_correction[:, 1] * wpix_out[1]  # Ignoring T
        beam_correction.append(sm_corr_P)
    beam_correction = np.array(beam_correction)

    # Would probably be better to use W but it's last dimension is a map, which makes things ill defined
    return np.einsum("fc, fl, fk->ckl", A, beam_correction, A)


def get_effective_common_beam(config: Config, A):
    lmax_convolution = 3 * config.nside
    wpix_out = hp.pixwin(
        config.nside, pol=True, lmax=lmax_convolution
    )  # Pixel window function of output maps
    Bl_gauss_common = hp.gauss_beam(
        np.radians(config.pre_proc_pars.common_beam_correction / 60),
        lmax=lmax_convolution,
        pol=True,
    )

    beam_P = Bl_gauss_common[:, 1] * wpix_out[1]  # Ignoring T

    beam_P_freq_array = np.array([beam_P for i in range(len(config.frequencies))])
    # Would probably be better to use W but it's last dimension is a map, which makes things ill defined
    return np.einsum("fc, fl, fk->ckl", A, beam_P_freq_array, A)


def initialize_nmt_workspace(
    nmt_bins, path_Cl_lens, nside, mask_analysis, effective_beam, purify_e, purify_b, n_iter
):
    Cl_lens = hp.read_cl(path_Cl_lens)
    map_T_init_wsp, map_Q_init_wsp, map_U_init_wsp = hp.synfast(Cl_lens, nside, new=True)

    fields_init_wsp = nmt.NmtField(
        mask_analysis,
        [map_Q_init_wsp, map_U_init_wsp],
        beam=effective_beam,
        purify_e=purify_e,
        purify_b=purify_b,
        n_iter=n_iter,
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

        fields = nmt.NmtField(
            mask_analysis,
            freq_noise_maps[f, 1:],
            beam=beam_f,
            purify_e=purify_e,
            purify_b=purify_b,
            n_iter=10,
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
