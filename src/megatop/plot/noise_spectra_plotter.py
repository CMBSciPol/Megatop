import argparse
from pathlib import Path

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np

from megatop import Config, DataManager
from megatop.utils import logger, mask
from megatop.utils.binning import load_nmt_binning
from megatop.utils.mock import get_Cl_CMB_model_from_manager
from megatop.utils.plot import plot_all_Cls, plot_all_Cls_diff


def plot_all_noise_spectra(manager, config):
    plot_dir = manager.path_to_spectra_plots
    plot_dir.mkdir(parents=True, exist_ok=True)

    binning_info = np.load(manager.path_to_binning, allow_pickle=True)
    bin_centre_lminlmax = binning_info["bin_centre_lminlmax"]
    bin_index_lminlmax = np.load(manager.path_to_binning, allow_pickle=True)["bin_index_lminlmax"]

    fig_EE, ax_EE = plt.subplots()
    fig_BB, ax_BB = plt.subplots()

    average_noise_CMB = np.zeros([4, len(bin_centre_lminlmax)])
    num_loaded_id = 0
    for id_sim in range(config.map_sim_pars.n_sim):
        try:
            fname_noise_Cls = manager.get_path_to_noise_spectra_cross_components(id_sim)
            all_noise_Cls = np.load(fname_noise_Cls, allow_pickle=True)
            all_noise_Cls_CMB = all_noise_Cls["Noise_CMBxNoise_CMB"][:, bin_index_lminlmax]

            average_noise_CMB += all_noise_Cls["Noise_CMBxNoise_CMB"][:, bin_index_lminlmax]

            ax_EE.plot(
                bin_centre_lminlmax,
                all_noise_Cls_CMB[0],
                label="Estimated Noise_CMB EE" if id_sim == 0 else None,
                linestyle="-",
                color="darkblue",
                alpha=0.2,
            )
            ax_BB.plot(
                bin_centre_lminlmax,
                all_noise_Cls_CMB[-1],
                label="Estimated Noise_CMB BB" if id_sim == 0 else None,
                linestyle="-",
                color="darkblue",
                alpha=0.2,
            )
            num_loaded_id += 1
        except FileNotFoundError:
            logger.warning(
                f"Noise Spectra file not found for id_sim={id_sim} at Path:{fname_noise_Cls}"
            )

    # WARNING: here we average the noise Nl (already averaged over noise sims) over the different sky sims
    # average_noise_CMB /= config.map_sim_pars.n_sim
    average_noise_CMB /= num_loaded_id

    ax_EE.plot(
        bin_centre_lminlmax,
        average_noise_CMB[0],
        label="Mean noise EE",
        color="brown",
        linestyle="-",
    )
    ax_BB.plot(
        bin_centre_lminlmax,
        average_noise_CMB[-1],
        label="Mean noise BB",
        color="brown",
        linestyle="-",
    )

    ax_EE.set_xlabel(r"$\ell$")
    ax_EE.set_ylabel(r"$C_{\ell}^{EE}$")
    ax_EE.legend()
    ax_EE.loglog()
    ax_EE.set_title("Noise_CMB EE spectra")
    fig_EE.savefig(plot_dir / "allskysims_NoiseCMB_EE_spectra.png")

    ax_BB.set_xlabel(r"$\ell$")
    ax_BB.set_ylabel(r"$C_{\ell}^{BB}$")
    ax_BB.legend()
    ax_BB.loglog()
    ax_BB.set_title("Noise_CMB BB spectra")
    fig_BB.savefig(plot_dir / "allskysims_NoiseCMB_BB_spectra.png")
    # closing figures
    plt.close(fig_EE)
    plt.close(fig_BB)


def plot_all_spectra(manager, config):
    plot_dir = manager.path_to_spectra_plots
    plot_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Saving plots to %s", plot_dir)

    logger.info("Saving plots to %s", plot_dir)

    binning_info = np.load(manager.path_to_binning, allow_pickle=True)
    bin_centre_lminlmax = binning_info["bin_centre_lminlmax"]
    bin_index_lminlmax = binning_info["bin_index_lminlmax"]

    nmt_bins = load_nmt_binning(manager)
    Cl_cmb_model = get_Cl_CMB_model_from_manager(manager)[:, : config.lmax + 1]

    bined_Cl_cmb_model = nmt_bins.bin_cell(Cl_cmb_model)[:, bin_index_lminlmax]
    bined_Cl_cmb_model_namaster = np.array(
        [bined_Cl_cmb_model[1], bined_Cl_cmb_model[4], bined_Cl_cmb_model[4], bined_Cl_cmb_model[2]]
    )
    del bined_Cl_cmb_model

    fig_EE, ax_EE = plt.subplots()
    fig_BB, ax_BB = plt.subplots()
    fig_EE_debiased, ax_EE_debiased = plt.subplots()
    fig_BB_debiased, ax_BB_debiased = plt.subplots()
    fig_EE_debiased_diff, ax_EE_debiased_diff = plt.subplots()
    fig_BB_debiased_diff, ax_BB_debiased_diff = plt.subplots()
    fig_EE_debiased, ax_EE_debiased = plt.subplots()
    fig_BB_debiased, ax_BB_debiased = plt.subplots()
    fig_EE_debiased_diff, ax_EE_debiased_diff = plt.subplots()
    fig_BB_debiased_diff, ax_BB_debiased_diff = plt.subplots()

    fig_summary, ax_summary = plt.subplots(3, 4, figsize=(24, 16))

    average_noise_CMB = np.zeros([4, len(bin_centre_lminlmax)])
    array_debiased_diff_model = np.zeros([config.map_sim_pars.n_sim, 4, len(bin_centre_lminlmax)])
    num_loaded_id = 0
    for id_sim in range(config.map_sim_pars.n_sim):
        try:
            fname_noise_Cls = manager.get_path_to_noise_spectra_cross_components(id_sim)
            fname_all_Cls = manager.get_path_to_spectra_cross_components(id_sim)

            all_noise_Cls = np.load(fname_noise_Cls, allow_pickle=True)

            all_Cls = np.load(fname_all_Cls, allow_pickle=True)

            cmb_cls = all_Cls["CMBxCMB"][:, bin_index_lminlmax]
            debiased_cmb_cls = (
                all_Cls["CMBxCMB"][:, bin_index_lminlmax]
                - all_noise_Cls["Noise_CMBxNoise_CMB"][:, bin_index_lminlmax]
            )
            average_noise_CMB += all_noise_Cls["Noise_CMBxNoise_CMB"][:, bin_index_lminlmax]

            ax_EE.plot(
                bin_centre_lminlmax,
                cmb_cls[0],
                label="Estimated CMB EE (noisy)" if id_sim == 0 else None,
                linestyle="-",
                color="darkblue",
                alpha=0.2,  # if not negative_bins_in_EE else 1.0,
            )
            ax_summary[0, 0].plot(
                bin_centre_lminlmax,
                cmb_cls[0],
                label="Estimated CMB EE (noisy)" if id_sim == 0 else None,
                linestyle="-",
                color="darkblue",
                alpha=0.2,  # if not negative_bins_in_EE else 1.0,
            )

            ax_BB.plot(
                bin_centre_lminlmax,
                cmb_cls[-1],
                label="Estimated CMB BB (noisy)" if id_sim == 0 else None,
                linestyle="-",
                color="darkblue",
                alpha=0.2,  # if not negative_bins_in_BB else 1.0,
            )
            ax_summary[0, -1].plot(
                bin_centre_lminlmax,
                cmb_cls[-1],
                label="Estimated CMB BB (noisy)" if id_sim == 0 else None,
                linestyle="-",
                color="darkblue",
                alpha=0.2,  # if not negative_bins_in_BB else 1.0,
            )

            ax_summary[0, 1].plot(
                bin_centre_lminlmax,
                cmb_cls[1],
                label="Estimated CMB EB (noisy)" if id_sim == 0 else None,
                linestyle="-",
                color="darkblue",
                alpha=0.2,  # if not negative_bins_in_BB else 1.0,
            )
            ax_summary[0, 2].plot(
                bin_centre_lminlmax,
                cmb_cls[2],
                label="Estimated CMB BE (noisy)" if id_sim == 0 else None,
                linestyle="-",
                color="darkblue",
                alpha=0.2,  # if not negative_bins_in_BB else 1.0,
            )

            ax_EE_debiased.plot(
                bin_centre_lminlmax,
                debiased_cmb_cls[0],
                label="Estimated CMB EE noise debiased" if id_sim == 0 else None,
                linestyle="-",
                color="darkblue",
                alpha=0.2,  # if not negative_bins_in_EE else 1.0,
            )
            ax_summary[1, 0].plot(
                bin_centre_lminlmax,
                debiased_cmb_cls[0],
                label="Estimated CMB EE noise debiased" if id_sim == 0 else None,
                linestyle="-",
                color="darkblue",
                alpha=0.2,  # if not negative_bins_in_EE else 1.0,
            )

            ax_BB_debiased.plot(
                bin_centre_lminlmax,
                debiased_cmb_cls[-1],
                label="Estimated CMB BB noise debiased" if id_sim == 0 else None,
                linestyle="-",
                color="darkblue",
                alpha=0.2,  # if not negative_bins_in_BB else 1.0,
            )
            ax_summary[1, -1].plot(
                bin_centre_lminlmax,
                debiased_cmb_cls[-1],
                label="Estimated CMB BB noise debiased" if id_sim == 0 else None,
                linestyle="-",
                color="darkblue",
                alpha=0.2,  # if not negative_bins_in_BB else 1.0,
            )
            ax_summary[1, 1].plot(
                bin_centre_lminlmax,
                debiased_cmb_cls[1],
                label="Estimated CMB EB noise debiased" if id_sim == 0 else None,
                linestyle="-",
                color="darkblue",
                alpha=0.2,  # if not negative_bins_in_BB else 1.0,
            )
            ax_summary[1, 2].plot(
                bin_centre_lminlmax,
                debiased_cmb_cls[2],
                label="Estimated CMB BE noise debiased" if id_sim == 0 else None,
                linestyle="-",
                color="darkblue",
                alpha=0.2,  # if not negative_bins_in_BB else 1.0,
            )

            # Just do one array god dammit
            diff_debiased_model = debiased_cmb_cls - bined_Cl_cmb_model_namaster
            # diff_debiased_model_EE = debiased_cmb_cls[0] - bined_Cl_cmb_model[1]
            # diff_debiased_model_EB = debiased_cmb_cls[1] - bined_Cl_cmb_model[3]
            # diff_debiased_model_BE = debiased_cmb_cls[2] - bined_Cl_cmb_model[4]
            # diff_debiased_model_BB = debiased_cmb_cls[-1] - bined_Cl_cmb_model[2]
            array_debiased_diff_model[id_sim] = diff_debiased_model
            # array_debiased_diff_model[id_sim, 0, :] = diff_debiased_model_EE
            # array_debiased_diff_model[id_sim, 1, :] = diff_debiased_model_BB
            ax_EE_debiased_diff.plot(
                bin_centre_lminlmax,
                # diff_debiased_model_EE,
                diff_debiased_model[0],
                label="Estimated CMB EE noise debiased - model" if id_sim == 0 else None,
                linestyle="-",
                color="darkblue",
                alpha=0.2,  # if not negative_bins_in_EE else 1.0,
            )
            ax_BB_debiased_diff.plot(
                bin_centre_lminlmax,
                diff_debiased_model[3],
                # diff_debiased_model_BB,
                label="Estimated CMB BB noise debiased - model" if id_sim == 0 else None,
                linestyle="-",
                color="darkblue",
                alpha=0.2,  # if not negative_bins_in_BB else 1.0,
            )
            label_namaster = ["EE", "EB", "BE", "BB"]
            for i in range(4):
                ax_summary[2, i].plot(
                    bin_centre_lminlmax,
                    diff_debiased_model[i],
                    label=f"Estimated CMB {label_namaster[i]} noise debiased - model"
                    if id_sim == 0
                    else None,
                    linestyle="-",
                    color="darkblue",
                    alpha=0.2,  # if not negative_bins_in_EE else 1.0,
                )

            negative_bins = cmb_cls[-1] < 0
            ax_BB.plot(
                bin_centre_lminlmax[negative_bins],
                np.abs(cmb_cls[-1][negative_bins]),
                label="ABS(Estimated CMB BB (noisy))" if id_sim == 0 else None,
                linestyle="--",
                color="green",
                alpha=0.2,
            )
            num_loaded_id += 1

        except FileNotFoundError:
            logger.warning(
                f"Noise Spectra OR All_Spectra file not found for id_sim={id_sim} at Path:{fname_noise_Cls} or {fname_all_Cls}"
            )

    # WARNING: here we average the noise Nl (already averaged over noise sims) over the different sky sims
    # average_noise_CMB /= config.map_sim_pars.n_sim
    average_noise_CMB /= num_loaded_id

    # noise_option = config.noise_sim_pars.noise_option
    # if noise_option == NoiseOption.NOISELESS:
    #     # TODO: this is a temporary fix, need to be done properly
    #     average_noise_CMB = np.zeros_like(average_noise_CMB)

    ax_EE.plot(
        bin_centre_lminlmax,
        bined_Cl_cmb_model_namaster[0] + average_noise_CMB[0],
        label="CMB EE model + mean noise",
        color="black",
        linestyle="--",
    )
    ax_BB.plot(
        bin_centre_lminlmax,
        bined_Cl_cmb_model_namaster[-1] + average_noise_CMB[-1],
        label="CMB BB model + mean noise",
        color="black",
        linestyle="--",
    )
    ax_EE.plot(
        bin_centre_lminlmax,
        average_noise_CMB[0],
        label="Mean noise EE",
        color="brown",
        linestyle="-",
    )
    ax_BB.plot(
        bin_centre_lminlmax,
        average_noise_CMB[-1],
        label="Mean noise BB",
        color="brown",
        linestyle="-",
    )

    ax_EE_debiased.plot(
        bin_centre_lminlmax,
        # bined_Cl_cmb_model[1],
        bined_Cl_cmb_model_namaster[0],
        label="CMB EE model",
        color="black",
        linestyle="--",
    )

    ax_BB_debiased.plot(
        bin_centre_lminlmax,
        # bined_Cl_cmb_model[2],
        bined_Cl_cmb_model_namaster[-1],
        label="CMB BB model",
        color="black",
        linestyle="--",
    )

    for i in range(4):
        ax_summary[0, i].plot(
            bin_centre_lminlmax,
            bined_Cl_cmb_model_namaster[i] + average_noise_CMB[i],
            label=f"CMB {label_namaster[i]} model + mean noise",
            color="black",
            linestyle="--",
        )
        # ax_summary[0,i].plot(
        #     bin_centre_lminlmax,
        #     average_noise_CMB[i],
        #     label=f"Mean noise {label_namaster[i]}",
        #     color="brown",
        #     linestyle="-",
        # )
        ax_summary[1, i].plot(
            bin_centre_lminlmax,
            bined_Cl_cmb_model_namaster[i],
            label=f"CMB {label_namaster[i]} model",
            color="black",
            linestyle="--",
        )
        for j in range(2):
            ax_summary[j, i].plot(
                bin_centre_lminlmax,
                average_noise_CMB[i],
                label=f"Mean noise {label_namaster[i]}",
                color="brown",
                linestyle="-",
            )
    mean_debiased_diff = np.mean(array_debiased_diff_model, axis=0)
    std_debiased_diff = np.std(array_debiased_diff_model, axis=0)
    # mean_debiased_diff_EE = np.mean(array_debiased_diff_model[:, 0, :], axis=0)
    # mean_debiased_diff_BB = np.mean(array_debiased_diff_model[:, 1, :], axis=0)
    # std_debiased_diff_EE = np.std(array_debiased_diff_model[:, 0, :], axis=0)
    # std_debiased_diff_BB = np.std(array_debiased_diff_model[:, 1, :], axis=0)

    analysis_mask = hp.read_map(manager.path_to_analysis_mask)
    fsky = mask.fsky_dof(analysis_mask)  # effective DOF for error bars (Hivon w2^2/w4)

    warning_labelEE = (
        "WARNING: negative bins in N_ell EE, use abs for error bars"
        if np.any(average_noise_CMB[0] < 0)
        else ""
    )
    warning_labelBB = (
        "WARNING: negative bins in N_ell BB, use abs for error bars"
        if np.any(average_noise_CMB[-1] < 0)
        else ""
    )

    cosmic_var_plus_noise = (bined_Cl_cmb_model_namaster + np.abs(average_noise_CMB)) * (
        2 / ((2 * bin_centre_lminlmax + 1) * config.map2cl_pars.delta_ell) / fsky
    ) ** 0.5

    # cosmic_var_plus_noise_EE = (bined_Cl_cmb_model_namaster[0] + np.abs(average_noise_CMB[0])) * (
    #     2 / ((2 * bin_centre_lminlmax + 1) * config.map2cl_pars.delta_ell) / fsky
    # ) ** 0.5
    # cosmic_var_plus_noise_BB = (bined_Cl_cmb_model_namaster[-1] + np.abs(average_noise_CMB[-1])) * (
    #     2 / ((2 * bin_centre_lminlmax + 1) * config.map2cl_pars.delta_ell) / fsky
    # ) ** 0.5

    ax_EE_debiased_diff.errorbar(
        bin_centre_lminlmax,
        # mean_debiased_diff_EE,
        mean_debiased_diff[0],
        # yerr=std_debiased_diff_EE,
        yerr=std_debiased_diff[0],
        label="Mean and stddev of CMB EE noise debiased - model",
        color="brown",
        linestyle="-",
        capsize=3,
    )
    ax_EE_debiased_diff.errorbar(
        bin_centre_lminlmax,
        # mean_debiased_diff_EE,
        mean_debiased_diff[0],
        # yerr=cosmic_var_plus_noise_EE,
        yerr=cosmic_var_plus_noise[0],
        label=r"<CMB EE noise debiased> - model $\pm (C_{\ell}+\langle N_{\ell}\rangle)\sqrt{\frac{2}{(2l+1)\Delta_{ell} f_{sky}}}$"
        + warning_labelEE,
        color="green",
        linestyle="-",
        capsize=3,
    )

    ax_BB_debiased_diff.errorbar(
        bin_centre_lminlmax,
        # mean_debiased_diff_BB,
        mean_debiased_diff[3],
        # yerr=std_debiased_diff_BB,
        yerr=std_debiased_diff[3],
        label="Mean and stddev of CMB BB noise debiased - model",
        color="brown",
        linestyle="-",
        capsize=3,
    )
    ax_BB_debiased_diff.errorbar(
        bin_centre_lminlmax,
        # mean_debiased_diff_BB,
        mean_debiased_diff[3],
        # yerr=cosmic_var_plus_noise_BB,
        yerr=cosmic_var_plus_noise[3],
        label=r"<CMB BB noise debiased> - model $\pm (C_{\ell}+\langle N_{\ell}\rangle)\sqrt{\frac{2}{(2l+1)\Delta_{ell} f_{sky}}}$"
        + warning_labelBB,
        color="green",
        linestyle="-",
        capsize=3,
    )

    for i in range(4):
        ax_summary[2, i].errorbar(
            bin_centre_lminlmax,
            mean_debiased_diff[i],
            yerr=std_debiased_diff[i],
            label=f"Mean and stddev of CMB {label_namaster[i]} noise debiased - model",
            color="brown",
            linestyle="-",
            capsize=3,
        )
        ax_summary[2, i].errorbar(
            bin_centre_lminlmax,
            mean_debiased_diff[i],
            yerr=cosmic_var_plus_noise[i],
            label=f"<CMB {label_namaster[i]} noise debiased> "
            + r"- model $\pm (C_{\ell}+\langle N_{\ell}\rangle)\sqrt{\frac{2}{(2l+1)\Delta_{ell} f_{sky}}}$"
            + warning_labelBB,
            color="green",
            linestyle="-",
            capsize=3,
        )

    ax_EE.set_xlabel(r"$\ell$")
    ax_EE.set_ylabel(r"$C_{\ell}^{EE}$")
    ax_EE.legend()
    ax_EE.loglog()
    ax_EE.set_title("CMB EE spectra")
    fig_EE.savefig(plot_dir / "allskysims_CMB_EE_spectra.png")

    ax_BB.set_xlabel(r"$\ell$")
    ax_BB.set_ylabel(r"$C_{\ell}^{BB}$")
    ax_BB.legend()
    ax_BB.loglog()
    ax_BB.set_title("CMB BB spectra")
    fig_BB.savefig(plot_dir / "allskysims_CMB_BB_spectra.png")

    ax_EE_debiased.set_xlabel(r"$\ell$")
    ax_EE_debiased.set_ylabel(r"$C_{\ell}^{EE}$")
    ax_EE_debiased.legend()
    ax_EE_debiased.loglog()
    ax_EE_debiased.set_title("CMB EE spectra")
    fig_EE_debiased.savefig(plot_dir / "allskysims_CMB_EE_debiased_spectra.png")

    ax_BB_debiased.set_xlabel(r"$\ell$")
    ax_BB_debiased.set_ylabel(r"$C_{\ell}^{BB}$")
    ax_BB_debiased.legend()
    ax_BB_debiased.loglog()
    ax_BB_debiased.set_title("CMB BB spectra")
    fig_BB_debiased.savefig(plot_dir / "allskysims_CMB_BB_debiased_spectra.png")

    lmax_cosmo = config.cl2r_pars.lmax_cosmo_analysis or config.lmax
    cosmo_mask = bin_centre_lminlmax <= lmax_cosmo
    # ylim_EE = 3.0 * np.nanmax(np.abs(cosmic_var_plus_noise_EE[cosmo_mask]))
    ylim_EE = 3.0 * np.nanmax(np.abs(cosmic_var_plus_noise[0][cosmo_mask]))
    # ylim_BB = 3.0 * np.nanmax(np.abs(cosmic_var_plus_noise_BB[cosmo_mask]))
    ylim_BB = 3.0 * np.nanmax(np.abs(cosmic_var_plus_noise[-1][cosmo_mask]))
    # ylim_all = 3.0 * np.nanmax(np.abs(cosmic_var_plus_noise[:, cosmo_mask]), axis=1)

    for ax, ylim, pol in [
        (ax_EE_debiased_diff, ylim_EE, "EE"),
        (ax_BB_debiased_diff, ylim_BB, "BB"),
    ]:
        ax.set_xlabel(r"$\ell$")
        ax.set_ylabel(rf"$C_{{\ell}}^{{{pol}}}$")
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.axvline(
            lmax_cosmo,
            color="gray",
            linestyle=":",
            linewidth=1,
            label=rf"$\ell_{{max}}^{{cosmo}}={lmax_cosmo}$",
        )
        ax.legend(fontsize=7)
        ax.set_title(rf"CMB {pol} spectra difference to model")
        ax.set_xscale("log")
        ax.set_ylim(-ylim, ylim)

    for i in range(4):
        ax_summary[2, i].axhline(0, color="black", linestyle="--", linewidth=1)
        ax_summary[j, i].set_title(rf"CMB {label_namaster[i]}")
        ax_summary[j, i].set_title(rf"CMB {label_namaster[i]} debiased")
        ax_summary[j, i].set_title(rf"CMB {label_namaster[i]} debiased - model")

        for j in range(3):
            ax_summary[j, i].set_xlabel(r"$\ell$")
            ax_summary[j, i].set_ylabel(rf"$C_{{\ell}}^{{{label_namaster[i]}}}$")
            ax_summary[j, i].axvline(
                lmax_cosmo,
                color="gray",
                linestyle=":",
                linewidth=1,
                label=rf"$\ell_{{max}}^{{cosmo}}={lmax_cosmo}$",
            )
            ax_summary[j, 2].legend(fontsize=7)
            ax_summary[j, i].set_xscale("log")
            ax_summary[j, i].set_yscale("linear" if j == 2 or i == 1 or i == 2 else "log")
            # if j == 2:
            #     # ylim = 3.0 * np.nanmax(np.abs(cosmic_var_plus_noise[i][cosmo_mask]))
            #     ax_summary[j,i].set_ylim(-ylim_all[i], ylim_all[i])

    fig_summary.tight_layout()
    fig_summary.savefig(plot_dir / "summary_all_spectra.png")
    plt.close(fig_summary)

    fig_EE_debiased_diff.savefig(plot_dir / "allskysims_CMB_EE_debiased_spectra_diff_to_model.png")
    fig_BB_debiased_diff.savefig(plot_dir / "allskysims_CMB_BB_debiased_spectra_diff_to_model.png")
    # closing figures
    plt.close(fig_EE)
    plt.close(fig_BB)
    plt.close(fig_EE_debiased)
    plt.close(fig_BB_debiased)
    plt.close(fig_EE_debiased_diff)
    plt.close(fig_BB_debiased_diff)


def plot_noise_spectra(manager, config, id_sim=None):
    plot_dir = manager.path_to_spectra_plots
    plot_dir.mkdir(parents=True, exist_ok=True)

    binning_info = np.load(manager.path_to_binning, allow_pickle=True)
    bin_centre_lminlmax = binning_info["bin_centre_lminlmax"]
    bin_index_lminlmax = np.load(manager.path_to_binning, allow_pickle=True)["bin_index_lminlmax"]
    try:
        fname_noise_Cls = manager.get_path_to_noise_spectra_cross_components(id_sim)
        all_noise_Cls_ = np.load(fname_noise_Cls, allow_pickle=True)
    except FileNotFoundError:
        logger.warning(
            f"Noise Spectra file not found for id_sim={id_sim} at Path:{fname_noise_Cls}, skipping plotting."
        )
        # TODO: try another id_sim if the file is not found for this one, instead of skipping plotting entirely?
        return
    all_noise_Cls_binranged = {}
    for k in all_noise_Cls_:
        all_noise_Cls_binranged[k] = all_noise_Cls_[k][:, bin_index_lminlmax]

    plot_all_Cls(
        all_noise_Cls_binranged,
        bin_centre_lminlmax,
        plot_dir,
        "noise_post_compsep_spectra",
        use_D_ell=False,
        y_axis_label=r"$C_{\ell}$",
    )

    fname_all_Cls = manager.get_path_to_spectra_cross_components(id_sim)
    all_Cls = np.load(fname_all_Cls, allow_pickle=True)

    debiased_cls = {}
    for key_cls, key_noise_cls in zip(all_Cls.keys(), all_noise_Cls_binranged.keys(), strict=False):
        debiased_cls[key_cls] = (
            all_Cls[key_cls][:, binning_info["bin_index_lminlmax"]]
            - all_noise_Cls_binranged[key_noise_cls]
        )

    plot_all_Cls(
        debiased_cls,
        bin_centre_lminlmax,
        plot_dir,
        "debiased_post_compsep_spectra",
        use_D_ell=False,
        y_axis_label=r"$C_{\ell}$",
    )

    nmt_bins = load_nmt_binning(manager)
    Cl_cmb_model = get_Cl_CMB_model_from_manager(manager)[:, : config.lmax + 1]

    bined_Cl_cmb_model = nmt_bins.bin_cell(Cl_cmb_model)[:, binning_info["bin_index_lminlmax"]]

    bined_Cl_cmb_model_dict = {
        "CMBxCMB": [
            bined_Cl_cmb_model[1],
            bined_Cl_cmb_model[4],
            bined_Cl_cmb_model[4],
            bined_Cl_cmb_model[2],
        ]
    }
    diabiased_cls_CMB_only = {"CMBxCMB": debiased_cls["CMBxCMB"]}

    plot_all_Cls_diff(
        diabiased_cls_CMB_only,
        bin_centre_lminlmax,
        bined_Cl_cmb_model_dict,
        plot_dir,
        "diff_debiased_CMB_spectra_vs_model",
        use_D_ell=False,
        y_axis_label=r"$C_{\ell}$",
    )

    cls_CMB_only = {"CMBxCMB": all_Cls["CMBxCMB"][:, binning_info["bin_index_lminlmax"]]}
    bined_biased_cl_cmb_model = {
        "CMBxCMB": bined_Cl_cmb_model_dict["CMBxCMB"]
        + all_noise_Cls_binranged["Noise_CMBxNoise_CMB"]
    }

    plot_all_Cls_diff(
        cls_CMB_only,
        bin_centre_lminlmax,
        bined_biased_cl_cmb_model,
        plot_dir,
        "diff_CMB_spectra_vs_biased_model",
        use_D_ell=False,
        y_axis_label=r"$C_{\ell}$",
    )


def main():
    parser = argparse.ArgumentParser(description="Plotter for map2cl output")
    parser.add_argument("--config", type=Path, help="config file")
    args = parser.parse_args()
    if args.config is None:
        logger.warning("No config file provided, using example config")
        config = Config.get_example()
    else:
        config = Config.load_yaml(args.config)
    manager = DataManager(config)
    manager.dump_config()

    logger.info("Plotting Noise spectra outputs...")

    n_sim_sky = config.map_sim_pars.n_sim
    if n_sim_sky == 0:
        id_sim = None
    else:
        logger.info("Plotting only simulation #0")
        id_sim = 0

    plot_noise_spectra(manager, config, id_sim=id_sim)

    if n_sim_sky != 0:
        logger.info("Plotting all spectra:")
        plot_all_spectra(manager, config)
        plot_all_noise_spectra(manager, config)


if __name__ == "__main__":
    main()
