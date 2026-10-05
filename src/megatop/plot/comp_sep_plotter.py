import argparse
from pathlib import Path

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

from megatop import Config, DataManager
from megatop.utils import Timer, logger
from megatop.utils.mask import apply_binary_mask
from megatop.utils.plot import freq_maps_plotter
import corner
from getdist import MCSamples, plots
from chainconsumer import ChainConsumer, Chain, Truth, PlotConfig

def get_angle_labels(config: Config) -> dict[str, str]:
    """Build angle_i -> 'EXP freq' labels dynamically from config.map_sets."""
    return {
        f"angle_{i}": f"{map_set.exp_tag} {map_set.freq_tag:g}"  # espace normal, pas de \
        for i, map_set in enumerate(config.map_sets)
    }

def clean_param_label(name: str, angle_labels: dict) -> str:
    if name in angle_labels:
        return angle_labels[name]
    if name == "beta_dust":
        return "β dust"      # espace au lieu de underscore
    if name == "beta_pl":
        return "β sync"
    return name

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
                    capsize=2,
                )
                ax_param.errorbar(
                    mean,
                    ylims[1] * 1.1,
                    xerr=std,
                    fmt="o",
                    color="darkblue",
                    label=r"$\langle\langle \text{chain} \rangle_{\rm step} \rangle_{sims} \pm \langle \sigma(\text{chain})_{\rm step} \rangle_{sims}$"
                    + legend_mean_of_std,
                    capsize=2,
                )
                ax_param.set_ylim([ylims[0], ylims[1] * 1.2])  # Extend y-limits for visibility
            # Update legend:
            ax_param.legend(
                loc="upper right", bbox_to_anchor=(1.0, 7), fontsize=10, frameon=True, fancybox=True
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
                        capsize=2,
                    )
                    ax_2d.errorbar(
                        mean_x,
                        mean_y,
                        xerr=std_x,
                        yerr=std_y,
                        fmt="o",
                        color="darkblue",
                        label=f"{param_name_x}, {param_name_y} mean ± std",
                        capsize=2,
                    )

def plot_hmc(manager: DataManager, config: Config, mc_samples, params_names=None):
    
    gd_plot = plots.get_subplot_plotter(width_inch=10)
    gd_plot.settings.figure_legend_frame = False
    gd_plot.settings.line_labels = False
    gd_plot.settings.alpha_filled_add = 0.5
    gd_plot.settings.alpha_factor_contour_lines = 0.5
    gd_plot.settings.axes_fontsize = 8
    gd_plot.settings.lab_fontsize = 8
    gd_plot.settings.num_plot_contours = 2

    plot_dir = manager.path_to_components_plots

    n = len(mc_samples)
    gd_plot.triangle_plot(
        mc_samples,
        filled=False,
        line_args=[{"lw": 1.0, "color": "darkblue", "alpha": 0.2}] * n,
        contour_colors=["darkblue"] * n,
        contour_args=[{"lw": 1.0, "color": "darkblue", "alpha": 0.2}] * n,
    )

    stats_params_dict = {}
    for name in params_names:
        mean_per_sim = np.array([np.mean(x.samples[:, x.index[name]]) for x in mc_samples])
        std_per_sim  = np.array([np.std(x.samples[:, x.index[name]])  for x in mc_samples])
        stats_params_dict[name] = {
            "mean":         np.mean(mean_per_sim),
            "mean_of_std":  np.mean(std_per_sim),
            "std_of_mean":  np.std(mean_per_sim),
            "mean_per_sim": mean_per_sim,
            "std_per_sim":  std_per_sim,
        }

    add_error_bars_to_getdist_plot(gd_plot, stats_params_dict)
    plt.savefig(plot_dir / Path("triangle_plot.pdf"), bbox_inches="tight")
    plt.show()

def corner_plot1(manager: DataManager, config: Config, cov_list, means_list, labels):

    angle_labels = get_angle_labels(config)
    n = cov_list[0].shape[0]
    if labels is None:
        labels = [f"x{i}" for i in range(n)]
    labels = [angle_labels.get(label, label) for label in labels]

    plot_dir = manager.path_to_components_plots
    for i, (cov, mean) in enumerate(zip(cov_list, means_list)):
        samples = np.random.multivariate_normal(mean, cov, size=100_000)
        corner.corner(samples, labels=labels, show_titles=True,
                      truths=mean,
                      quantiles=[0.16, 0.5, 0.84],
                      #smooth=1.0,
                      #smooth1d=4.0,
                      title_fmt=".6f",
                      title_kwargs={"fontsize": 12})
        plt.savefig(plot_dir / Path(f"corner_plot_sim_{i}.png"))
        plt.close()

def corner_plot2(manager: DataManager, config: Config, cov_list, means_list, labels):

    angle_labels = get_angle_labels(config)
    n = cov_list[0].shape[0]
    if labels is None:
        labels = [f"x{i}" for i in range(n)]
    labels = [clean_param_label(label, angle_labels) for label in labels]

    plot_dir = manager.path_to_components_plots
    for i, (cov, mean) in enumerate(zip(cov_list, means_list)):
        c = ChainConsumer()
        c.add_chain(Chain.from_covariance(
            mean,
            cov,
            columns=labels,
            name=f"sim_{i}",
            color="#1f77b4",
            shade=True,
            shade_alpha=0.6,
            kde=False,
            smooth=0,
        ))
        #c.add_truth(Truth(location=dict(zip(labels, mean))))
        c.add_truth(Truth(
            location=dict(zip(labels, mean)),
            line_style="--",
            line_width=0.7,
            color="black",
        ))

        c.set_plot_config(
            PlotConfig(
                label_font_size=10,
                tick_font_size=8,
                max_ticks=3,
                serif=False,
                usetex=False,
            )
        )

        fig = c.plotter.plot()

        # --- kill matplotlib's offset notation on every axis ---
        for ax in fig.get_axes():
            ax.xaxis.set_major_formatter(
                matplotlib.ticker.ScalarFormatter(useOffset=False, useMathText=False)
            )
            ax.yaxis.set_major_formatter(
                matplotlib.ticker.ScalarFormatter(useOffset=False, useMathText=False)
            )
            ax.xaxis.get_major_formatter().set_scientific(True)
            ax.yaxis.get_major_formatter().set_scientific(True)
            ax.xaxis.get_major_formatter().set_powerlimits((-3, 3))
            ax.yaxis.get_major_formatter().set_powerlimits((-3, 3))
        # --- end fix ---

        # --- strip ChainConsumer's auto-appended "[scale]" from axis labels only ---
        import re
        for ax in fig.get_axes():
            for get_label, set_label in [
                (ax.get_xlabel, ax.set_xlabel),
                (ax.get_ylabel, ax.set_ylabel),
            ]:
                lbl = get_label()
                if lbl:
                    set_label(re.sub(r"\s*\[.*?\]\s*$", "", lbl))
        # --- end fix ---

        fig.savefig(plot_dir / Path(f"corner_plot_sim_{i}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

def plot_compsep(manager: DataManager, config: Config, id_sim: int | None = None):
    plot_dir = manager.path_to_components_plots
    plot_dir.mkdir(parents=True, exist_ok=True)

    fname_compmaps = manager.get_path_to_components_maps(id_sim)
    comp_maps = np.load(fname_compmaps)

    experiments = set(m.exp_tag for m in config.map_sets)
    binary_mask = np.zeros(hp.nside2npix(config.nside))
    for exp in experiments:
        binary_mask = np.maximum(binary_mask, hp.read_map(manager.path_to_binary_mask(exp)))
    comp_maps = apply_binary_mask(comp_maps, binary_mask, unseen=True)

    freq_maps_plotter(
        config,
        np.array([comp_maps[0]]),
        plot_dir,
        "CMB_post_compsep_maps",
        component="CMB post-compsep",
    )
    freq_maps_plotter(
        config,
        np.array([comp_maps[1]]),
        plot_dir,
        "dust_post_compsep_maps",
        component="Dust post-compsep",
    )
    if config.parametric_sep_pars.include_synchrotron:
        freq_maps_plotter(
            config,
            np.array([comp_maps[2]]),
            plot_dir,
            "synch_post_compsep_maps",
            component="Synch post-compsep",
        )
    # --- True noise of this sky sim, propagated through the compsep (NOUVEAU) ---
    fname_compmaps = Path(fname_compmaps)
    fname_true_noise = fname_compmaps.with_name(fname_compmaps.stem + "_true_noise.npy")
    if not fname_true_noise.exists():
        logger.warning(f"True noise component maps not found at {fname_true_noise}, skipping.")
        return

    noise_maps = np.load(fname_true_noise)
    noise_maps = apply_binary_mask(noise_maps, binary_mask, unseen=True)

    comp_names = ["CMB", "dust", "synch"]
    for c in range(noise_maps.shape[0]):
        freq_maps_plotter(
            config,
            np.array([noise_maps[c]]),
            plot_dir,
            f"{comp_names[c]}_true_noise_post_compsep_maps",
            component=f"{comp_names[c]} true noise post-compsep",
        )


def plot_compsep_stats(manager: DataManager, config: Config):
    if config.map_sim_pars.n_sim == 0 or config.map_sim_pars.n_sim is None:
        logger.info("No sky simulations, skipping component separation statistics plotter.")
        return

    compsep_results_params = []
    param_res_list = []
    cov = []
    mc_samples = []
    convergence_count = 0
    for sky_sims_id in range(config.map_sim_pars.n_sim):
        fname_compsepresults = manager.get_path_to_compsep_results(id_sim=sky_sims_id)
        compsep_results = np.load(fname_compsepresults, allow_pickle=True)
        params_names = compsep_results["params_names"]
        params = compsep_results["x"]
        # "params" is saved in the same order as "x" (unlike "params_names", which is
        # reordered to angles-first by megabuster's Hessienne_plus_corner_plot). Use it
        # to realign the data columns to the params_names ordering below.
        x_order_names = compsep_results["params"]
        if config.parametric_sep_pars.megabuster_options.use_hessienne:
            cov.append(np.load(fname_compsepresults, allow_pickle=True)["Cov"])
        if config.parametric_sep_pars.megabuster_options.use_hmc:
            mc_samples.append(np.load(fname_compsepresults, allow_pickle=True)["mc_samples"].item())

        convergence = compsep_results["success"].astype(bool)
        print('convergence', convergence)
        #if convergence:
        # "x"/"params" order isn't stable across sims, so reorder this sim's row to
        # match params_names right away instead of reordering the whole stacked
        # array afterwards (which would silently mix up columns across sims).
        reorder_idx = [list(x_order_names).index(name) for name in params_names]
        params = params[reorder_idx]
        compsep_results_params.append(params)
        param_res_list.append(params)
        convergence_count += 1
    param_res_list = np.array(param_res_list)
    compsep_results_params = np.array(compsep_results_params)

    # Angles are stored in radians in "x", but Cov (from the Hessian) is already in
    # degrees for angle entries. Convert the angle columns (and their std) to degrees
    # to match, so values/uncertainties are consistent and human-readable.
    angle_col_idx = [i for i, name in enumerate(params_names) if str(name).startswith("angle")]
    compsep_results_params[:, angle_col_idx] *= 180.0 / np.pi
    param_res_list[:, angle_col_idx] *= 180.0 / np.pi

    logger.info(
        f"Component sepatation converged successfully for of {100 * convergence_count / config.map_sim_pars.n_sim:.2f}% the maps."
    )

    plot_dir = manager.path_to_components_plots

    compsep_results_last = np.load(fname_compsepresults, allow_pickle=True)

    # Plotting histograms of result parameters:
    fig, axes = plt.subplots(1, params_names.shape[0], figsize=(2.6*len(params_names), 5))
    axes = np.atleast_1d(axes)

    label_map = {
        "Dust.beta_d": r"$\beta_{\rm dust}$",
        "Synchrotron.beta_pl": r"$\beta_{\rm sync}$",
        **get_angle_labels(config),
    }

    for i, (ax, param_name) in enumerate(zip(axes, params_names, strict=False)):
        data = compsep_results_params[:, i]

        ax.hist(data, bins=25, histtype="step", density=False, color="darkblue")

        mean_param = np.mean(data)
        std_param = np.std(data)

        ax.axvline(mean_param, color="mediumvioletred", linestyle="-", linewidth=1.5)

        is_angle = str(param_name).startswith("angle")
        unit_suffix = r"^\circ" if is_angle else ""
        display_label = label_map.get(param_name, param_name)
        title_label = display_label.replace("_", r"\_") if is_angle else display_label.strip("$")

        ax.grid(True, linestyle="--", color="lightgrey", alpha=0.7)
        ax.set_xlabel(display_label + (" [deg]" if is_angle else ""), fontsize=9)
        ax.set_ylabel("Counts", fontsize=9)
        ax.tick_params(axis="both", labelsize=7)

        # Titre sur deux lignes + police réduite pour éviter le chevauchement
        title = (
            rf"${title_label}$" + "\n"
            + rf"${mean_param:.3f}{unit_suffix} \pm {std_param:.5f}{unit_suffix}$"
        )
        ax.set_title(title, fontsize=8)

    plt.tight_layout()
    plt.savefig(plot_dir / Path("statistics_compsep.png"), dpi=150, bbox_inches="tight")
    plt.close()

    #Plot compsep corner plot using Hessienne of HMC according to config file
    if config.parametric_sep_pars.megabuster_options.use_hessienne:
        corner_plot2(manager, config, cov, param_res_list, params_names)
    if config.parametric_sep_pars.megabuster_options.use_hmc:
        plot_hmc(manager, config, mc_samples, params_names)

    return


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
        logger.info("Plotting only simulation #0")
        id_sim = 0

    logger.info("Plotting comp sep outputs...")
    with Timer("comp-sep-plotter"):
        plot_compsep(manager, config, id_sim=id_sim)
        plot_compsep_stats(manager, config)


if __name__ == "__main__":
    main()