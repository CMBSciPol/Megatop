import argparse
from pathlib import Path

import numpy as np
from getdist import MCSamples, plots
from matplotlib import pyplot as plt

from megatop import Config, DataManager
from megatop.pipeline.cl2r_estimater import Cl_CMB_model, compute_generic_Cl
from megatop.utils import logger


def plot_all_cornerplots(manager: DataManager, config: Config):
    """Plot corner plot resulting from the MCMC for all sky realizations."""
    # Load true parameters:
    r_sim = config.map_sim_pars.r_input
    A_lens_sim = config.map_sim_pars.A_lens
    n_sim_sky = config.map_sim_pars.n_sim

    all_samples = []
    colors = [
        plt.cm.plasma(i / (n_sim_sky - 1)) for i in range(n_sim_sky)
    ]  # Convert to list of color values

    for id_sim in range(n_sim_sky):
        fname_chains = manager.get_path_to_mcmc_chains(sub=id_sim)
        mcmc = np.load(fname_chains, allow_pickle=True)
        chains = mcmc["mcmc_chains"]
        param_names = mcmc["param_names"]

        samples = MCSamples(samples=chains, names=param_names, labels=param_names)
        all_samples.append(samples)

    # Make plot:
    gd_plot = plots.get_subplot_plotter(width_inch=8)
    gd_plot.settings.figure_legend_frame = False
    gd_plot.settings.alpha_filled_add = 0.4
    gd_plot.settings.line_styles = ["-"] * n_sim_sky

    gd_plot.triangle_plot(
        all_samples,
        filled=True,
        legend_loc=None,
        line_args=[{"lw": 1.0, "color": colors[i]} for i in range(n_sim_sky)],
        contour_colors=colors,
        markers={"r": r_sim, "A_{lens}": A_lens_sim},
    )

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

    fname_chains = manager.get_path_to_mcmc_chains(sub=id_sim)
    mcmc = np.load(fname_chains, allow_pickle=True)
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
        markers={"r": r_sim, "A_{lens}": A_lens_sim},
    )

    # Save figure:
    plot_dir = manager.path_to_mcmc_plots
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_name = f"corner_plot_skysim{id_sim}"
    plt.savefig(plot_dir / plot_name, bbox_inches="tight")
    plt.clf()


def plot_spectra_comparison(manager: DataManager, config: Config, id_sim: int | None = None):
    # Load necessary parameters
    dust_marg = config.cl2r_pars.dust_marg
    sync_marg = config.cl2r_pars.sync_marg
    lmin = config.general_pars.lmin
    lmax = config.general_pars.lmax
    sky_model = "".join(config.map_sim_pars.sky_model)

    # Load spectra data
    Cl_CMBxCMB_BB_est = np.load(manager.get_path_to_spectra_cross_components(sub=id_sim))[
        "CMBxCMB"
    ][3][1:]
    Cl_DustxDust_BB_est = np.load(manager.get_path_to_spectra_cross_components(sub=id_sim))[
        "DustxDust"
    ][3][1:]
    Nl_CMBxCMB_BB_est = np.load(manager.get_path_to_noise_spectra_cross_components(sub=id_sim))[
        "Noise_CMBxNoise_CMB"
    ][3][1:]

    ls_bins_low = np.load(manager.get_path_to_spectra_binning(sub=id_sim))["bin_low"][1:]
    ls_bins_high = np.load(manager.get_path_to_spectra_binning(sub=id_sim))["bin_high"][1:]
    ls_bins_lminlmax_idx = np.load(manager.get_path_to_spectra_binning(sub=id_sim))[
        "bin_index_lminlmax"
    ][1:]
    ls_bins_lminlmax_centre = np.load(manager.get_path_to_spectra_binning(sub=id_sim))[
        "bin_centre_lminlmax"
    ][1:]

    Cl_BB_prim_generic, Cl_BB_lensing_generic = compute_generic_Cl(lmin, lmax)

    # Load estimated parameters from MCMC chains
    fname_chains = manager.get_path_to_mcmc_chains(sub=id_sim)
    mcmc = np.load(fname_chains, allow_pickle=True)
    chains = mcmc["mcmc_chains"]
    param_names = mcmc["param_names"]
    samples = MCSamples(samples=chains, names=param_names, labels=param_names)
    theta_est = samples.getMeans()

    if not dust_marg and not sync_marg:
        r_est, A_lens_est = theta_est
    if dust_marg and not sync_marg:
        r_est, A_lens_est, A_dust_est = theta_est
    if not dust_marg and sync_marg:
        r_est, A_lens_est, A_sync_est = theta_est
    if dust_marg and sync_marg:
        r_est, A_lens_est, A_dust_est, A_sync_est = theta_est

    # Compute individual Cl components
    Cl_BB_prim_est = r_est * Cl_BB_prim_generic
    Cl_BB_lensing_est = A_lens_est * Cl_BB_lensing_generic
    if dust_marg:
        Cl_BB_dust_est = A_dust_est * Cl_DustxDust_BB_est

    # Compute model spectrum based on estimated parameters
    Cl_CMBxCMB_BB_model = Cl_CMB_model(
        theta_est,
        dust_marg,
        sync_marg,
        lmin,
        Cl_BB_prim_generic,
        Cl_BB_lensing_generic,
        Cl_DustxDust_BB_est,
        Nl_CMBxCMB_BB_est,
        ls_bins_low,
        ls_bins_high,
        ls_bins_lminlmax_idx,
    )

    # Plot spectra
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(
        ls_bins_lminlmax_centre,
        Cl_CMBxCMB_BB_est,
        marker=".",
        ls="--",
        lw="0.75",
        label=r"$\hat{C}_\ell^{CMB}$",
        color="darkblue",
        zorder=1,
    )
    ax.plot(
        ls_bins_lminlmax_centre,
        Cl_DustxDust_BB_est,
        marker=".",
        ls="--",
        lw="0.75",
        label=r"$\hat{C}_\ell^{dust}$",
        color="mediumvioletred",
        zorder=1,
    )
    ax.plot(
        ls_bins_lminlmax_centre,
        Nl_CMBxCMB_BB_est,
        marker=".",
        ls="--",
        lw="0.75",
        label=r"$\hat{N}_\ell$",
        color="hotpink",
        zorder=2,
    )
    ax.plot(
        np.arange(lmin, lmax + 1),
        Cl_BB_prim_est,
        label=r"$C_\ell^{prim, \rm est} = r^{\rm est} \cdot C_\ell^{\rm prim}(r=1)$ ",
        color="darkcyan",
        linestyle="-",
        lw="0.75",
    )

    ax.plot(
        np.arange(lmin, lmax + 1),
        Cl_BB_lensing_est,
        label=r"$C_\ell^{lensing, \rm est} = A_{\rm lens}^{\rm est} \cdot C_\ell^{\rm prim}(r=0)$",
        color="seagreen",
        linestyle="-",
        lw="0.75",
    )
    """
    ax.plot(
        np.arange(lmin, lmax + 1),
        Cl_BB_prim_generic,
        label="Cl_prim_sim",
        color="crimson",
        linestyle="-",
        lw="0.75"
    )
    """
    if dust_marg:
        ax.plot(
            ls_bins_lminlmax_centre,
            Cl_BB_dust_est,
            label=r"$C_\ell^{dust, \rm est} = A_{\rm dust}^{\rm est} \cdot \hat{C}_\ell^{\rm dust}$",
            color="darkseagreen",
            linestyle="-",
            lw="0.75",
        )

    ax.plot(
        ls_bins_lminlmax_centre,
        Cl_CMBxCMB_BB_model,
        marker="x",
        lw=0.75,
        label=r"$C_\ell^{CMB, \rm est} + \hat{N}_{\ell}$",
        color="cornflowerblue",
        zorder=2,
    )

    for l_bin in ls_bins_lminlmax_centre:
        ax.axvline(l_bin, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\ell$", fontsize=14)
    ax.set_ylabel(r"$C_\ell^{BB} (\mu{\rm K}^2)$", fontsize=14)
    ax.set_title(f"sky model: {sky_model}", loc="left", fontsize=12)
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))

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
        plot_all_cornerplots(manager, config)
        id_sim = 1
        logger.info(f"Plotting for sky simulation #{id_sim}")
        plot_single_cornerplot(manager, config, id_sim=id_sim)
        plot_spectra_comparison(manager, config, id_sim=id_sim)


if __name__ == "__main__":
    main()
