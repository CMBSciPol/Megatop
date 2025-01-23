import argparse
import os

import healpy as hp
import IPython
import numpy as np
import pymaster as nmt
import scipy
from matplotlib import gridspec

from megatop.utils import utils
from megatop.utils.metadata_manager import BBmeta


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


def spectra_estimation(meta):
    meta.timer.start("loading_comp_maps")
    fname_comp_maps = os.path.join(meta.components_directory, "components_maps.npy")
    comp_maps = np.load(fname_comp_maps)
    fname_invAtNA = os.path.join(meta.components_directory, "invAtNA.npy")
    invAtNA = np.load(fname_invAtNA)
    meta.timer.stop("loading_comp_maps", meta.logger, "Loading component maps")

    # Creating/loading bins

    bin_low, bin_high, bin_centre = utils.create_binning(
        meta.nside, meta.map2cl_pars["delta_ell"], end_first_bin=meta.lmin
    )

    bin_index_lminlmax = np.where((bin_low >= meta.lmin) & (bin_high <= meta.lmax))[0]

    np.savez(
        os.path.join(meta.spectra_directory, "binning.npz"),
        bin_low=bin_low,
        bin_high=bin_high,
        bin_centre=bin_centre,
        bin_index_lminlmax=bin_index_lminlmax,
    )
    nmt_bins = nmt.NmtBin.from_edges(bin_low, bin_high + 1)
    # b = nmt.NmtBin.from_nside_linear(meta.nside, nlb=int(meta.map2cl_pars['delta_ell']))

    # Loading analysis mask
    mask_analysis = meta.read_mask("analysis")
    binary_mask = meta.read_mask("binary").astype(bool)

    # Initializin workspace
    meta.timer.start("initializing_workspace")
    path_Cl_lens = meta.get_fname_cls_fiducial_cmb("lensed")
    Cl_lens = hp.read_cl(path_Cl_lens)

    wpix_out = hp.pixwin(
        meta.general_pars["nside"], pol=True, lmax=3 * meta.nside
    )  # Pixel window function of output maps
    Bl_gauss_common = hp.gauss_beam(
        np.radians(meta.pre_proc_pars["common_beam_correction"] / 60), lmax=3 * meta.nside, pol=True
    )
    wpix_in = hp.pixwin(
        meta.general_pars["nside"], pol=True, lmax=3 * meta.nside
    )  # Pixel window function of input maps
    wpix_in[1][0:2] = 1.0  # in order not to divide by 0

    effective_beam = Bl_gauss_common[:, 1] * wpix_out[1]  # / wpix_in[1]
    map_T_init_wsp, map_Q_init_wsp, map_U_init_wsp = hp.synfast(Cl_lens, meta.nside, new=True)
    fields_init_wsp = nmt.NmtField(
        mask_analysis,
        [map_Q_init_wsp, map_U_init_wsp],
        beam=effective_beam,
        purify_e=meta.map2cl_pars["purify_e"],
        purify_b=meta.map2cl_pars["purify_b"],
        n_iter=meta.map2cl_pars["n_iter_namaster"],
    )
    workspace_cc = nmt.NmtWorkspace()
    IPython.embed()
    workspace_cc.compute_coupling_matrix(
        fields_init_wsp, fields_init_wsp, nmt_bins, n_iter=meta.map2cl_pars["n_iter_namaster"]
    )
    meta.timer.stop("initializing_workspace", meta.logger, "Initializing workspace")
    # TODO: update namaster and test from_fields for workspace (see SOOPERCOOL)
    # workspaceff = nmt.NmtWorkspace.from_fields(fields_init_wsp,fields_init_wsp,nmt_bins)

    # Testing the function
    meta.timer.start("spectra_estimation")

    # if hp.UNSEEN is used in comp-sep, the comp-maps will use it as well which will be a problem for namaster, we regularize it here
    comp_maps *= binary_mask
    # The noise map outputed by comp-sep is not cleanly masked.
    # To avoid numerical issues, we apply the mask to the noise maps.

    noise_map_after_compsep = np.load(
        os.path.join(meta.components_directory, "noise_map_after_compsep.npy")
    )
    noise_map_after_compsep[..., np.where(binary_mask == 0)[0]] = 0
    invAtNA[..., np.where(binary_mask == 0)[0]] = 0
    # Let's comput the matrix sqrt of invAtNA:

    # sqrt_invAtNA = scipy.linalg.sqrtm(invAtNA) this doesn't work because of the shape
    meta.timer.start("sqrtm_invAtNA")
    sqrt_invAtNA = np.zeros_like(invAtNA)
    for p in range(invAtNA.shape[-1]):
        if binary_mask[p] == 0:
            continue
        for stokes in range(invAtNA.shape[-2]):
            sqrt_invAtNA[..., stokes, p] = scipy.linalg.sqrtm(invAtNA[..., stokes, p])
    meta.timer.stop("sqrtm_invAtNA", meta.logger, "Computing matrix sqrt of invAtNA")

    # IPython.embed()
    if meta.parametric_sep_pars["DEBUG_UseSynchrotron"]:
        comp_dict = {"CMB": comp_maps[0], "Dust": comp_maps[1], "Synch": comp_maps[2]}
        # noise_dict = {'NoiseCMB': invAtNA[0,0], 'NoiseDust': invAtNA[1,1], 'NoiseSynch': invAtNA[2,2]}
        # noise_dict = {'NoiseCMB': sqrt_invAtNA[0,0], 'NoiseDust': sqrt_invAtNA[1,1], 'NoiseSynch': sqrt_invAtNA[2,2]}
        noise_dict = {
            "NoiseCMB": noise_map_after_compsep[0],
            "NoiseDust": noise_map_after_compsep[1],
            "NoiseSynch": noise_map_after_compsep[2],
        }
        # noise_dict = {'NoiseCMB': np.sqrt(invAtNA[0,0]), 'NoiseDust': np.sqrt(invAtNA[1,1]), 'NoiseSynch': np.sqrt(invAtNA[2,2])}
        noise_dict_offdiag = {
            "NoiseCMBDust": invAtNA[0, 1],
            "NoiseDustSynch": invAtNA[1, 2],
            "NoiseCMBSynch": invAtNA[0, 2],
        }
    else:
        comp_dict = {"CMB": comp_maps[0], "Dust": comp_maps[1]}
        # noise_dict = {'NoiseCMB': invAtNA[0,0], 'NoiseDust': invAtNA[1,1]}
        noise_dict = {"NoiseCMB": sqrt_invAtNA[0, 0], "NoiseDust": sqrt_invAtNA[1, 1]}
        # noise_dict = {'NoiseCMB': np.sqrt(invAtNA[0,0]), 'NoiseDust': np.sqrt(invAtNA[1,1])}
        noise_dict_offdiag = {"NoiseCMBDust": invAtNA[0, 1]}
    all_Cls = compute_auto_cross_cl_from_maps_list(
        comp_dict,
        mask_analysis,
        effective_beam,
        workspace_cc,
        purify_e=meta.map2cl_pars["purify_e"],
        purify_b=meta.map2cl_pars["purify_b"],
        n_iter=meta.map2cl_pars["n_iter_namaster"],
    )

    np.savez(os.path.join(meta.spectra_directory, "cross_components_Cls.npz"), **all_Cls)

    meta.timer.stop("spectra_estimation", meta.logger, "Spectra estimation")

    meta.timer.start("noise_spectra_estimation")

    # Here we assume that InvAtNA is symmetric, which seems true up to numerical precision
    Cls_noise = compute_auto_cross_cl_from_maps_list(
        noise_dict,
        mask_analysis,
        effective_beam,
        workspace_cc,
        purify_e=meta.map2cl_pars["purify_e"],
        purify_b=meta.map2cl_pars["purify_b"],
        n_iter=meta.map2cl_pars["n_iter_namaster"],
    )
    np.savez(os.path.join(meta.spectra_directory, "noise_Cls.npz"), **Cls_noise)

    Cls_noise_offdiag = compute_auto_cross_cl_from_maps_list(
        noise_dict_offdiag,
        mask_analysis,
        effective_beam,
        workspace_cc,
        purify_e=meta.map2cl_pars["purify_e"],
        purify_b=meta.map2cl_pars["purify_b"],
        n_iter=meta.map2cl_pars["n_iter_namaster"],
    )
    np.savez(os.path.join(meta.spectra_directory, "noise_Cls_offdiag.npz"), **Cls_noise_offdiag)

    meta.timer.stop("noise_spectra_estimation", meta.logger, "Noise spectra estimation")

    return all_Cls


def main():
    parser = argparse.ArgumentParser(description="simplistic simulator")  # TODO change name ??
    parser.add_argument("--globals", type=str, help="Path to yaml with global parameters")
    args = parser.parse_args()
    meta = BBmeta(args.globals)
    res = spectra_estimation(meta)
