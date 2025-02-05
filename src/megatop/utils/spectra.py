import healpy as hp
import numpy as np
import pymaster as nmt

from megatop import Config


def compute_auto_cross_cl_from_maps_list(
    maps_dict, mask, beam, workspace, purify_e=True, purify_b=True, n_iter=3
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
    for i, f_a in enumerate(fields):
        for j, f_b in enumerate(fields):
            if i <= j:
                cl_coupled = nmt.compute_coupled_cell(f_a, f_b)
                cl_decoupled = workspace.decouple_cell(cl_coupled)
                cl_list.append(cl_decoupled)

    # Store in dictionary with key x key
    cl_dict = {}
    for i, key in enumerate(maps_dict.keys()):
        for j, key2 in enumerate(maps_dict.keys()):
            if i <= j:
                cl_dict[key + "x" + key2] = cl_list.pop(0)

    return cl_dict


def get_common_beam_wpix(common_beam_fwhm_arcmin, nside):
    wpix_out = hp.pixwin(nside, pol=True, lmax=3 * nside)  # Pixel window function of output maps
    Bl_gauss_common = hp.gauss_beam(
        np.radians(common_beam_fwhm_arcmin / 60), lmax=3 * nside, pol=True
    )

    return Bl_gauss_common[:, 1] * wpix_out[1]


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
