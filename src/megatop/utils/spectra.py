import healpy as hp
import numpy as np
import pymaster as nmt


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
