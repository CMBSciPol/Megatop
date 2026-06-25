import argparse
from pathlib import Path

import healpy as hp
import numpy as np
from getdist import MCSamples, plots
from matplotlib import pyplot as plt

from megatop import Config, DataManager
from megatop.config import NoiseOption
from megatop.pipeline.cl2r_estimater import Cl_CMB_model, compute_generic_Cl
from megatop.plot.r_stats_plotter import get_params_statistics
from megatop.utils import logger
from megatop.utils.binning import load_nmt_binning


def add_error_bars_to_getdist_plot(gd_plot, stats_params_dict):
    """Adds error bars to the GetDist plot based on the statistics dictionary."""

    legend_mean_of_std = ""
    legend_std_of_mean = ""
    for param_name, stats in stats_params_dict.items():
        mean = stats["mean"]
        std = stats["mean_of_std"]
        std_of_mean = stats["std_of_mean"]
        legend_mean_of_std += "\n" + rf"${param_name}$ = {mean:.4f}" + r"$\pm$" + f"{std:.4f}"
        legend_std_of_mean += (
            "\n" + rf"${param_name}$ = {mean:.4f}" + r"$\pm$" + f"{std_of_mean:.4f}"
        )
    # 1D plots:
    for param_name, stats in stats_params_dict.items():
        mean = stats["mean"]
        std = stats["mean_of_std"]
        std_of_mean = stats["std_of_mean"]
        ax_param = gd_plot.get_axes_for_params(param_name)
        ylims = ax_param.get_ylim()

        ax_param.errorbar(
            mean,
            ylims[1] * 1.1,
            xerr=std_of_mean,
            fmt="o",
            color="darkgreen",
            label=r"$\langle\langle \text{chain} \rangle_{\rm step} \rangle_{sims} \pm \sigma(\langle \text{chain} \rangle_{\rm step})_{sims}$"
            + legend_std_of_mean,
            capsize=4,
        )
        ax_param.errorbar(
            mean,
            ylims[1] * 1.1,
            xerr=std,
            fmt="o",
            color="darkblue",
            label=r"$\langle\langle \text{chain} \rangle_{\rm step} \rangle_{sims} \pm \langle \sigma(\text{chain})_{\rm step} \rangle_{sims}$"
            + legend_mean_of_std,
            capsize=4,
        )
        ax_param.set_ylim([ylims[0], ylims[1] * 1.2])  # Extend y-limits for visibility
    # Update legend:
    ax_param.legend(
        loc="lower right", bbox_to_anchor=(1.0, 1.5), fontsize=10, frameon=True, fancybox=True
    )
    # 2D plots:
    for param_name_x, stats_x in stats_params_dict.items():
        for param_name_y, stats_y in stats_params_dict.items():
            ax_2d = gd_plot.get_axes_for_params(param_name_x, param_name_y)
            if ax_2d is None:
                continue

            mean_x = stats_x["mean"]
            mean_y = stats_y["mean"]
            std_x = stats_x["mean_of_std"]
            std_y = stats_y["mean_of_std"]
            std_of_mean_x = stats_x["std_of_mean"]
            std_of_mean_y = stats_y["std_of_mean"]

            ax_2d.errorbar(
                mean_x,
                mean_y,
                xerr=std_of_mean_x,
                yerr=std_of_mean_y,
                fmt="o",
                color="darkgreen",
                label=f"{param_name_x}, {param_name_y} mean ± std of mean",
                capsize=4,
            )
            ax_2d.errorbar(
                mean_x,
                mean_y,
                xerr=std_x,
                yerr=std_y,
                fmt="o",
                color="darkblue",
                label=f"{param_name_x}, {param_name_y} mean ± std",
                capsize=4,
            )


def plot_all_cornerplots(manager: DataManager, config: Config):
    """Plot corner plot resulting from the MCMC for all sky realizations."""
    # Load true parameters:
    r_sim = config.map_sim_pars.r_input
    A_lens_sim = config.map_sim_pars.A_lens
    Birefringence = config.map_sim_pars.Birefringence
    n_sim_sky = config.map_sim_pars.n_sim

    all_samples = []

    # colors = (
    #     [plt.cm.plasma(0.5)]
    #     if n_sim_sky == 1
    #     else [plt.cm.plasma(i / (n_sim_sky - 1)) for i in range(n_sim_sky)]
    # )

    for id_sim in range(n_sim_sky):
        try:
            # TODO: cleaner test, flag?
            fname_chains = manager.get_path_to_mcmc_chains(id_sim)
            # binning_info = np.load(manager.path_to_binning, allow_pickle=True)
            # ls_bins_lminlmax_idx = binning_info["bin_index_lminlmax"]
            # Cl_CMBxCMB_BB_est = np.load(manager.get_path_to_spectra_cross_components(id_sim))["CMBxCMB"][3][ls_bins_lminlmax_idx]
            # Nl_CMBxCMB_BB_est = np.load(manager.get_path_to_noise_spectra_cross_components(id_sim))["Noise_CMBxNoise_CMB"][3][ls_bins_lminlmax_idx]
            # if np.any(Cl_CMBxCMB_BB_est<0) or np.any(Nl_CMBxCMB_BB_est<0):
            #     logger.error(f"negative bins in Cl CMB or Nl, skipping id_sim={id_sim} in plot_all_cornerplots")
            # else:
            mcmc = np.load(fname_chains, allow_pickle=True)
            chains = mcmc["mcmc_chains"]
            param_names = mcmc["param_names"]

            samples = MCSamples(
                samples=chains,
                names=param_names,
                labels=param_names,
            )
            all_samples.append(samples)
        except FileNotFoundError:
            logger.warning(f"MCMC chain file not found for id_sim={id_sim} at Path:{fname_chains}")

    # Make plot:
    gd_plot = plots.get_subplot_plotter(width_inch=8)
    gd_plot.settings.figure_legend_frame = False
    gd_plot.settings.line_labels = False
    gd_plot.settings.alpha_filled_add = 0.4
    gd_plot.settings.alpha_factor_contour_lines = 0.4
    gd_plot.settings.line_styles = ["-"] * n_sim_sky

    gd_plot.triangle_plot(
        all_samples,
        filled=False,
        legend_loc=None,
        legend_labels=[None] * n_sim_sky,
        line_args=[{"lw": 1.0, "color": "darkblue", "alpha": 0.4} for i in range(n_sim_sky)],
        # line_args=[{"lw": 1.0, "color": colors[i]} for i in range(n_sim_sky)],
        contour_colors=["darkblue"] * n_sim_sky,
        contour_args=[{"lw": 1.0, "color": "darkblue", "alpha": 0.4} for i in range(n_sim_sky)],
        # contour_colors=colors,
        markers={"r": r_sim, "A_{lens}": A_lens_sim,"Birefringence": Birefringence},
    )

    stats_params_dict = get_params_statistics(manager, config)

    add_error_bars_to_getdist_plot(gd_plot, stats_params_dict)

    # Save figure:
    plot_dir = manager.path_to_mcmc_plots
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_name = "corner_plot_all_skysims"
    plt.savefig(plot_dir / plot_name, bbox_inches="tight")
    plt.clf()


def plot_single_cornerplot(manager: DataManager, config: Config, id_sim: int | None = None):
    """Plot corner plot resulting from the MCMC for a single sky realization."""
    # Load parameters and mcmc chains:
    r_sim = config.map_sim_pars.r_input
    A_lens_sim = config.map_sim_pars.A_lens
    Birefringence = config.map_sim_pars.Birefringence

    try:
        fname_chains = manager.get_path_to_mcmc_chains(id_sim)
        mcmc = np.load(fname_chains, allow_pickle=True)
    except FileNotFoundError:
        logger.error(
            f"MCMC chain file not found for id_sim={id_sim} at Path:{fname_chains}, skipping plot_single_cornerplot"
        )
        return
    chains = mcmc["mcmc_chains"]
    param_names = mcmc["param_names"]

    # Make plot:
    samples = MCSamples(samples=chains, names=param_names, labels=param_names)

    gd_plot = plots.get_subplot_plotter(width_inch=8)
    gd_plot.settings.figure_legend_frame = False
    gd_plot.settings.alpha_filled_add = 0.4

    gd_plot.triangle_plot(
        [samples],
        filled=True,
        legend_loc="upper right",
        line_args=[{"lw": 1.5, "color": "darkblue"}],
        contour_colors=["darkblue"],
        markers={"r": r_sim, "A_{lens}": A_lens_sim,"Birefringence": Birefringence},
    )

    # Save figure:
    plot_dir = manager.path_to_mcmc_plots
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_name = f"corner_plot_skysim{id_sim}"
    plt.savefig(plot_dir / plot_name, bbox_inches="tight")
    plt.clf()


def plot_spectra_comparison(manager: DataManager, config: Config, id_sim: int | None = None):
    dust_marg = config.cl2r_pars.dust_marg
    sync_marg = config.cl2r_pars.sync_marg
    sky_model = "".join(config.map_sim_pars.sky_model)
    binning_info = np.load(manager.path_to_binning, allow_pickle=True)
    ls_bins_lminlmax_idx = binning_info["bin_index_lminlmax"]

    # Load spectra data
    try:
        Cl_CMBxCMB_BB_est = np.load(manager.get_path_to_spectra_cross_components(id_sim))["CMBxCMB"][3]
        Cl_CMBxCMB_EE_est = np.load(manager.get_path_to_spectra_cross_components(id_sim))["CMBxCMB"][0]
        Cl_CMBxCMB_EB_est = np.load(manager.get_path_to_spectra_cross_components(id_sim))["CMBxCMB"][1]
        Cl_DustxDust_BB_est = np.load(manager.get_path_to_spectra_cross_components(id_sim))["DustxDust"][3]
    except FileNotFoundError:
        logger.error(
            f"Spectrum file not found for id_sim={id_sim}, for paths {manager.get_path_to_spectra_cross_components(id_sim)}, skipping plot_spectra_comparison"
        )
        return

    all_noise_options = [
        config.noise_sim_pars.experiments[map_set.exp_tag].noise_option
        for map_set in config.map_sets
    ]

    if not np.all(np.array(all_noise_options) == NoiseOption.NOISELESS):
        Nl_CMBxCMB_BB_est = np.zeros_like(Cl_CMBxCMB_BB_est)
        Nl_CMBxCMB_EE_est = np.zeros_like(Cl_CMBxCMB_EE_est)
        Nl_CMBxCMB_EB_est = np.zeros_like(Cl_CMBxCMB_EB_est)
    else:
        try:
            Nl_CMBxCMB_BB_est = np.load(manager.get_path_to_noise_spectra_cross_components(id_sim))["Noise_CMBxNoise_CMB"][3]
            Nl_CMBxCMB_EE_est = np.load(manager.get_path_to_noise_spectra_cross_components(id_sim))["Noise_CMBxNoise_CMB"][0]
            Nl_CMBxCMB_EB_est = np.load(manager.get_path_to_noise_spectra_cross_components(id_sim))["Noise_CMBxNoise_CMB"][1]
        except FileNotFoundError:
            logger.error(
            f"Spectrum file not found for id_sim={id_sim}, for paths {manager.get_path_to_spectra_cross_components(id_sim)}, skipping plot_spectra_comparison"
            )
            return

    nmt_bins = load_nmt_binning(manager)
    ls_bins_lminlmax_centre = binning_info["bin_centre_lminlmax"]

    if config.cl2r_pars.load_model_spectra:
        Cl_BB_lensing_generic = hp.read_cl(manager.path_to_lensed_scalar)[2][
            : config.lmax + 1
        ]
        Cl_EE_generic         = hp.read_cl(manager.path_to_lensed_scalar)[1][
            : config.lmax + 1
        ]
        Cl_BB_prim_generic    = hp.read_cl(manager.path_to_unlensed_scalar_tensor_r1)[2][
            : config.lmax + 1
        ]
    else:
        Cl_EE_generic, Cl_BB_prim_generic, Cl_BB_lensing_generic = compute_generic_Cl(0, 3 * config.nside - 1)

    # Load MCMC chains
    fname_chains = manager.get_path_to_mcmc_chains(id_sim)
    mcmc = np.load(fname_chains, allow_pickle=True)
    chains = mcmc["mcmc_chains"]
    param_names = mcmc["param_names"]
    samples = MCSamples(samples=chains, names=param_names, labels=param_names)
    theta_est = samples.getMeans()

    if not dust_marg and not sync_marg:
        r_est, A_lens_est, biref_est = theta_est
    if dust_marg and not sync_marg:
        r_est, A_lens_est, biref_est, A_dust_est = theta_est
    if not dust_marg and sync_marg:
        r_est, A_lens_est, biref_est, A_sync_est = theta_est
    if dust_marg and sync_marg:
        r_est, A_lens_est, biref_est, A_dust_est, A_sync_est = theta_est

    biref_rad = (np.pi / 180) * biref_est  # degrés -> radians

    # Composantes CMB intrinseques (avant birefringence)
    Cl_BB_prim_est    = r_est * Cl_BB_prim_generic
    Cl_BB_lensing_est = A_lens_est * Cl_BB_lensing_generic
    Cl_BB_CMB_est     = Cl_BB_prim_est + Cl_BB_lensing_est
    if dust_marg:
        Cl_BB_dust_est = A_dust_est * Cl_DustxDust_BB_est

    # Mélange E->B induit par la biréfringence (sur la grille ell complète)
    Cl_BB_biref_mixing = np.sin(2 * biref_rad)**2 * Cl_EE_generic
    Cl_BB_obs_nobins   = np.cos(2 * biref_rad)**2 * Cl_BB_CMB_est + Cl_BB_biref_mixing
    Cl_EB_obs_nobins   = 0.5 * np.sin(4 * biref_rad) * (Cl_EE_generic - Cl_BB_CMB_est)

    # Modele total binnee (ce que la vraisemblance utilise)
    Cl_model_matrix = Cl_CMB_model(
        theta_est,
        dust_marg,
        sync_marg,
        Cl_EE_generic,
        Cl_BB_prim_generic,
        Cl_BB_lensing_generic,
        Cl_DustxDust_BB_est,
        Nl_CMBxCMB_EE_est,
        Nl_CMBxCMB_BB_est,
        Nl_CMBxCMB_EB_est,
        Nl_CMBxCMB_EB_est,  # BE = EB
        ls_bins_lminlmax_idx,
        nmt_bins,
    )
    # Cl_CMB_model retourne C[2,2,n_bins] -> extraire BB et EB
    Cl_CMBxCMB_BB_model = Cl_model_matrix[1, 1]
    Cl_CMBxCMB_EB_model = Cl_model_matrix[0, 1]

    # ------------------------------------------------------------------ Plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # ---- Panneau haut : BB ----
    ax = axes[0]

    ax.plot(ls_bins_lminlmax_centre, Cl_CMBxCMB_BB_est,
            marker=".", ls="--", lw=0.75,
            label=r"$\hat{C}_\ell^{BB,\,\rm CMB}$ (données)", color="darkblue", zorder=1)
    ax.plot(ls_bins_lminlmax_centre, Cl_DustxDust_BB_est,
            marker=".", ls="--", lw=0.75,
            label=r"$\hat{C}_\ell^{BB,\,\rm dust}$", color="mediumvioletred", zorder=1)
    ax.plot(ls_bins_lminlmax_centre, Nl_CMBxCMB_BB_est,
            marker=".", ls="--", lw=0.75,
            label=r"$\hat{N}_\ell^{BB}$", color="hotpink", zorder=2)

    ells = np.arange(len(Cl_BB_prim_est))
    ax.plot(ells, Cl_BB_prim_est,
            label=r"$r^{\rm est}\,C_\ell^{\rm prim}(r=1)$, $r=$" + f"{r_est:.4f}",
            color="darkcyan", lw=0.75)
    ax.plot(ells, Cl_BB_lensing_est,
            label=r"$A_{\rm lens}^{\rm est}\,C_\ell^{\rm lens}$, $A_{\rm lens}=$" + f"{A_lens_est:.3f}",
            color="seagreen", lw=0.75)
    ax.plot(ells, Cl_BB_biref_mixing,
            label=r"$\sin^2(2\beta)\,C_\ell^{EE}$ (fuite $E\!\to\!B$), $\beta=$" + f"{biref_est:.3f}°",
            color="orange", lw=0.75, linestyle="-.")
    ax.plot(ells, Cl_BB_obs_nobins,
            label=r"$C_\ell^{BB,\rm obs}$ (biréfringence, non binné)",
            color="goldenrod", lw=0.75, linestyle=":")

    if dust_marg:
        ax.plot(ls_bins_lminlmax_centre, Cl_BB_dust_est,
                label=r"$A_{\rm dust}^{\rm est}\,\hat{C}_\ell^{\rm dust}$",
                color="darkseagreen", lw=0.75)

    ax.plot(ls_bins_lminlmax_centre, Cl_CMBxCMB_BB_model,
            marker="x", lw=0.75,
            label=r"Modèle total $C_\ell^{BB,\rm mod} + \hat{N}_\ell^{BB}$",
            color="cornflowerblue", zorder=2)

    for l_bin in ls_bins_lminlmax_centre:
        ax.axvline(l_bin, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

    ax.set_yscale("log")
    ax.set_ylabel(r"$C_\ell^{BB}\ (\mu{\rm K}^2)$", fontsize=13)
    ax.set_title(f"sky model: {sky_model}", loc="left", fontsize=12)
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=9)

    # ---- Panneau bas : EB ----
    ax = axes[1]

    ax.plot(ls_bins_lminlmax_centre, Cl_CMBxCMB_EB_est,
            marker=".", ls="--", lw=0.75,
            label=r"$\hat{C}_\ell^{EB,\,\rm CMB}$ (données)", color="darkblue", zorder=1)
    ax.plot(ls_bins_lminlmax_centre, Nl_CMBxCMB_EB_est,
            marker=".", ls="--", lw=0.75,
            label=r"$\hat{N}_\ell^{EB}$", color="hotpink", zorder=2)
    ax.plot(ells, Cl_EB_obs_nobins,
            label=r"$\frac{1}{2}\sin(4\beta)\,(C_\ell^{EE}-C_\ell^{BB})$ (non binné)",
            color="orange", lw=0.75, linestyle="-.")
    ax.plot(ls_bins_lminlmax_centre, Cl_CMBxCMB_EB_model,
            marker="x", lw=0.75,
            label=r"Modèle total $C_\ell^{EB,\rm mod} + \hat{N}_\ell^{EB}$",
            color="cornflowerblue", zorder=2)

    for l_bin in ls_bins_lminlmax_centre:
        ax.axvline(l_bin, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

    ax.set_xlabel(r"$\ell$", fontsize=13)
    ax.set_ylabel(r"$C_\ell^{EB}\ (\mu{\rm K}^2)$", fontsize=13)
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=9)

    plt.tight_layout()
    plot_dir = manager.path_to_mcmc_plots
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_dir / f"spectra_comparison_skysim{id_sim}.png", bbox_inches="tight")
    plt.clf()


def main():
    parser = argparse.ArgumentParser(description="Plotter for component separation output")
    parser.add_argument("--config", type=Path, help="config file")
    args = parser.parse_args()
    if args.config is None:
        logger.warning("No config file provided, using example config")
        config = Config.get_example()
    else:
        config = Config.load_yaml(args.config)
    manager = DataManager(config)
    manager.dump_config()

    n_sim_sky = config.map_sim_pars.n_sim
    if n_sim_sky == 0:
        id_sim = None
    else:
        id_sim = 0
        logger.info(f"Plotting for sky simulation #{id_sim}")
        plot_single_cornerplot(manager, config, id_sim=id_sim)
        plot_spectra_comparison(manager, config, id_sim=id_sim)
        plot_all_cornerplots(manager, config)


if __name__ == "__main__":
    main()
