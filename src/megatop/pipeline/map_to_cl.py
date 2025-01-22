import argparse
import os

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import pymaster as nmt
import scipy
from matplotlib import gridspec

from megatop.utils import BBmeta, Timer, utils


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


def plot_all_Cls(all_Cls, bin_centre, file_path, cmb_theory_cls=None):
    # Define the labels
    labels = ["EE", "EB", "BE", "BB"]

    # Create the figure
    fig, axs = plt.subplots(4, 1, figsize=(10, 20))

    axs = axs.flatten()

    # Loop over the different power spectra
    for i, label in enumerate(labels):
        # Loop over the different components
        for j, key in enumerate(all_Cls.keys()):
            axs[i].plot(bin_centre, all_Cls[key][i], label=key, color="C" + str(j))
            axs[i].plot(bin_centre, -all_Cls[key][i], linestyle="--", color="C" + str(j))
        if cmb_theory_cls is not None:
            axs[i].plot(bin_centre, cmb_theory_cls[i], label="Input CMB", color="black")

        axs[i].set_title(label)
        axs[i].set_xlabel(r"$\ell$")
        axs[i].set_ylabel(r"$C_{\ell}$")
        axs[i].legend()
        axs[i].set_yscale("log")
        axs[i].set_xscale("log")

    plt.savefig(file_path)
    plt.close()


# def plot_all_Cls_and_diffs(all_Cls, reference_cl, bin_centre, file_path, reference_name='Input CMB'):

#     # Define the labels
#     labels = ['EE', 'EB', 'BE', 'BB']

#     # Create the figure
#     fig, axs = plt.subplots(8, 1, figsize=(10, 20), gridspec_kw={'height_ratios': [2, 1, 2, 1, 2, 1, 2, 1]}, sharex=True,
#                             )
#     axs = axs.flatten()

#     # Loop over the different power spectra
#     for i in range(axs.size):
#         # Loop over the different components
#         for j, key in enumerate(all_Cls.keys()):
#             if not i % 2: # Every other plot shows the spectra, the other the difference wrt the reference (else case)
#                 axs[i].plot(bin_centre, all_Cls[key][int(i/2)], label=key, color='C' + str(j))
#                 axs[i].plot(bin_centre, -all_Cls[key][int(i/2)], linestyle='--', color='C' + str(j))
#                 axs[i].plot(bin_centre, reference_cl[int(i/2)], label=reference_name, color='black')

#                 axs[i].set_title(labels[int(i/2)])
#                 axs[i].set_xlabel(r'$\ell$')
#                 axs[i].set_ylabel(r'$C_{\ell}$')
#                 if i == 0:
#                     axs[i].legend()

#                 axs[i].set_yscale('log')
#                 axs[i].set_xscale('log')

#             else:
#                 if key=='CMBxCMB':
#                     axs[i].plot(bin_centre, all_Cls[key][int(i/2)] - reference_cl[int(i/2)], label=key, color='C' + str(j))
#                     xlims = axs[i].get_xlim()
#                     axs[i].hlines(0, xlims[0], xlims[1], color='black', linestyle='--')
#                     axs[i].set_xlim(xlims)
#                     axs[i].set_xlabel(r'$\ell$')
#                     axs[i].set_ylabel(r'$\Delta C_{\ell}$')
#                     axs[i].set_xscale('log')

#     plt.savefig(file_path)
#     plt.close()


def plot_all_Cls_and_diffs(
    all_Cls, reference_cl, bin_centre, file_path, reference_name="Input CMB"
):
    # Define the labels
    labels = ["EE", "EB", "BE", "BB"]

    # Set up figure and gridspec
    fig = plt.figure(figsize=(6, 16))
    # 8 rows total: 2 for each plot pair + 2 for spacing
    gs = gridspec.GridSpec(11, 1, figure=fig, height_ratios=[1, 1, 0.6, 1, 1, 0.6, 1, 1, 0.6, 1, 1])

    # Plot each pair with shared x-axis and no space between
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[3])
    ax4 = fig.add_subplot(gs[4], sharex=ax3)
    ax5 = fig.add_subplot(gs[6])
    ax6 = fig.add_subplot(gs[7], sharex=ax5)
    ax7 = fig.add_subplot(gs[9])
    ax8 = fig.add_subplot(gs[10], sharex=ax7)

    # Hide x-axis labels for the top plots of each pair
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax3.get_xticklabels(), visible=False)
    plt.setp(ax5.get_xticklabels(), visible=False)

    # Reduce spacing between plots
    gs.update(hspace=0)  # No space between subplots within pairs

    # Add spacing between the pairs
    fig.subplots_adjust(hspace=0.5)  # Space only between pairs

    for i, (top_ax, bot_ax) in enumerate(
        zip([ax1, ax3, ax5, ax7], [ax2, ax4, ax6, ax8], strict=False)
    ):
        for j, key in enumerate(all_Cls.keys()):
            top_ax.plot(bin_centre, all_Cls[key][i], label=key, color="C" + str(j))
            top_ax.plot(bin_centre, -all_Cls[key][i], linestyle="--", color="C" + str(j))
            top_ax.plot(bin_centre, reference_cl[i], label=reference_name, color="black")

            top_ax.set_title(labels[i])
            top_ax.set_xlabel(r"$\ell$")
            top_ax.set_ylabel(r"$C_{\ell}$")
            if i == 0:
                top_ax.legend()

            top_ax.set_yscale("log")
            top_ax.set_xscale("log")

            if key == "CMBxCMB":
                bot_ax.plot(
                    bin_centre, all_Cls[key][i] - reference_cl[i], label=key, color="C" + str(j)
                )
                xlims = bot_ax.get_xlim()
                bot_ax.hlines(0, xlims[0], xlims[1], color="black", linestyle="--")
                bot_ax.set_xlim(xlims)
                bot_ax.set_xlabel(r"$\ell$")
                bot_ax.set_ylabel(r"$\Delta C_{\ell}$")
                bot_ax.set_xscale("log")

    plt.savefig(file_path)
    plt.close()


def spectra_estimation(args):
    meta = BBmeta(args.globals)
    timer_spectra = Timer()

    timer_spectra.start("loading_comp_maps")
    fname_comp_maps = os.path.join(meta.components_directory, "components_maps.npy")
    comp_maps = np.load(fname_comp_maps)
    fname_invAtNA = os.path.join(meta.components_directory, "invAtNA.npy")
    invAtNA = np.load(fname_invAtNA)
    timer_spectra.stop("loading_comp_maps", "Loading component maps", args.verbose)

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
    timer_spectra.start("initializing_workspace")
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
    workspace_cc.compute_coupling_matrix(
        fields_init_wsp, fields_init_wsp, nmt_bins, n_iter=meta.map2cl_pars["n_iter_namaster"]
    )
    timer_spectra.stop("initializing_workspace", "Initializing workspace", args.verbose)
    # TODO: update namaster and test from_fields for workspace (see SOOPERCOOL)
    # workspaceff = nmt.NmtWorkspace.from_fields(fields_init_wsp,fields_init_wsp,nmt_bins)

    # Testing the function
    timer_spectra.start("spectra_estimation")

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
    timer_spectra.start("sqrtm_invAtNA")
    sqrt_invAtNA = np.zeros_like(invAtNA)
    for p in range(invAtNA.shape[-1]):
        if binary_mask[p] == 0:
            continue
        for stokes in range(invAtNA.shape[-2]):
            sqrt_invAtNA[..., stokes, p] = scipy.linalg.sqrtm(invAtNA[..., stokes, p])
    timer_spectra.stop("sqrtm_invAtNA", "Computing matrix sqrt of invAtNA", args.verbose)

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

    # if args.plots:
    #     all_Clslminlmax = utils.apply_lminlmax_to_dict(all_Cls, bin_index_lminlmax)
    timer_spectra.stop("spectra_estimation", "Spectra estimation", args.verbose)

    timer_spectra.start("noise_spectra_estimation")

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
    # IPython.embed()
    # Cls_noise_sqrt = compute_auto_cross_cl_from_maps_list(sqrt_noise_dict, mask_analysis, effective_beam, workspace_cc, purify_e=meta.map2cl_pars['purify_e'],
    #                                                purify_b=meta.map2cl_pars['purify_b'], n_iter=meta.map2cl_pars['n_iter_namaster'])
    # if args.plots: Cls_noiselminlmax = utils.apply_lminlmax_to_dict(Cls_noise, bin_index_lminlmax)

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
    # if args.plots: Cls_noise_offdiaglminlmax = utils.apply_lminlmax_to_dict(Cls_noise_offdiag, bin_index_lminlmax)

    timer_spectra.stop("noise_spectra_estimation", "Noise spectra estimation", args.verbose)

    if args.plots:
        print("WARNING: Plots are now done in plot_spectra.py and not in map_to_cl.py")

    # if False:  # args.plots:
    #     timer_spectra.start("plotting")
    #     plot_dir = meta.plot_dir_from_output_dir(meta.spectra_directory_rel)
    #     input_cmb_spectra = utils.get_Cl_CMB_model_from_meta(meta)[0][:, : 3 * meta.nside]
    #     binned_input_cmb_spectra = nmt_bins.bin_cell(input_cmb_spectra)
    #     reshape_input_cmb_spectra = np.array(
    #         [
    #             binned_input_cmb_spectra[1],
    #             binned_input_cmb_spectra[-2],
    #             binned_input_cmb_spectra[-2],
    #             binned_input_cmb_spectra[2],
    #         ]
    #     )

    #     plot_all_Cls(
    #         all_Clslminlmax,
    #         bin_centre[bin_index_lminlmax],
    #         plot_dir + "/all_Cls.png",
    #         cmb_theory_cls=reshape_input_cmb_spectra[:, bin_index_lminlmax],
    #     )
    #     plot_all_Cls_and_diffs(
    #         {"CMBxCMB": all_Clslminlmax["CMBxCMB"]},
    #         reshape_input_cmb_spectra[:, bin_index_lminlmax],
    #         bin_centre[bin_index_lminlmax],
    #         plot_dir + "/Delta_CMB.png",
    #         reference_name="Input CMB",
    #     )

    #     unbiased_Cls = {}
    #     for key_signal, key_noise in zip(all_Cls.keys(), Cls_noise.keys()):
    #         unbiased_Cls[key_signal] = all_Clslminlmax[key_signal] - Cls_noiselminlmax[key_noise]

    #     plot_all_Cls(
    #         unbiased_Cls,
    #         bin_centre[bin_index_lminlmax],
    #         plot_dir + "/unbiased_Cls.png",
    #         cmb_theory_cls=reshape_input_cmb_spectra[:, bin_index_lminlmax],
    #     )
    #     plot_all_Cls_and_diffs(
    #         {"CMBxCMB": unbiased_Cls["CMBxCMB"]},
    #         reshape_input_cmb_spectra[:, bin_index_lminlmax],
    #         bin_centre[bin_index_lminlmax],
    #         plot_dir + "/Delta_CMB_debiased.png",
    #         reference_name="Input CMB",
    #     )

    #     plot_all_Cls(
    #         Cls_noiselminlmax,
    #         bin_centre[bin_index_lminlmax],
    #         plot_dir + "/Noise_Cls.png",
    #         cmb_theory_cls=reshape_input_cmb_spectra[:, bin_index_lminlmax],
    #     )
    #     plot_all_Cls(
    #         Cls_noise_offdiaglminlmax,
    #         bin_centre[bin_index_lminlmax],
    #         plot_dir + "/Noise_Cls_offdiag.png",
    #         cmb_theory_cls=reshape_input_cmb_spectra[:, bin_index_lminlmax],
    #     )
    #     timer_spectra.stop("plotting", "Plotting", args.verbose)

    return all_Cls


def main():
    parser = argparse.ArgumentParser(description="simplistic simulator")  # TODO change name ??
    parser.add_argument("--globals", type=str, help="Path to yaml with global parameters")
    parser.add_argument("--plots", action="store_true", help="Plot the generated maps if True.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    _ = spectra_estimation(args)


if __name__ == "__main__":
    main()
