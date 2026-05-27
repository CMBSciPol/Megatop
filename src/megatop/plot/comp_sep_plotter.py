import argparse
from pathlib import Path

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import corner

from megatop import Config, DataManager
from megatop.utils import Timer, logger
from megatop.utils.mask import apply_binary_mask
from megatop.utils.plot import freq_maps_plotter
from getdist import MCSamples, plots


def plot_compsep(manager: DataManager, config: Config, id_sim: int | None = None):
    plot_dir = manager.path_to_components_plots
    plot_dir.mkdir(parents=True, exist_ok=True)

    try:
        fname_compmaps = manager.get_path_to_components_maps(sub=id_sim)
        comp_maps = np.load(fname_compmaps)
    except FileNotFoundError:
        logger.warning(
            f"Component separation maps for simulation {id_sim} not found, skipping plotting."
        )
        return

    binary_mask = hp.read_map(manager.path_to_binary_mask)
    comp_maps = apply_binary_mask(comp_maps, binary_mask, unseen=True)

    freq_maps_plotter(
        config,
        np.array([comp_maps[0]]),
        plot_dir,
        f"CMB_post_compsep_maps_{id_sim}",
        component="CMB post-compsep",
    )
    freq_maps_plotter(
        config,
        np.array([comp_maps[1]]),
        plot_dir,
        f"dust_post_compsep_maps_{id_sim}",
        component="Dust post-compsep",
    )
    if config.parametric_sep_pars.include_synchrotron:
        freq_maps_plotter(
            config,
            np.array([comp_maps[2]]),
            plot_dir,
            f"synch_post_compsep_maps_{id_sim}",
            component="Synch post-compsep",
        )

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

    print('mc_sample ?',type(mc_samples[0]))

    n = len(mc_samples)
    gd_plot.triangle_plot(
        mc_samples,
        filled=False,
        line_args=[{"lw": 1.0, "color": "darkblue", "alpha": 0.2}] * n,
        contour_colors=["darkblue"] * n,
        contour_args=[{"lw": 1.0, "color": "darkblue", "alpha": 0.2}] * n,
    )

    print('params_names ?', params_names)

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

def corner_plot(manager: DataManager, config: Config, cov_list, means_list, labels):

    n = cov_list[0].shape[0]
    if labels is None:
        labels = [f"x{i}" for i in range(n)]
    
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

def plot_compsep_stats(manager: DataManager, config: Config):
    if config.map_sim_pars.n_sim == 0 or config.map_sim_pars.n_sim is None:
        logger.info("No sky simulations, skipping component separation statistics plotter.")
        return

    param_res_list = []
    cov = []
    mc_samples = []
    for sky_sims_id in range(config.map_sim_pars.n_sim):
        try:
            fname_compsepresults = manager.get_path_to_compsep_results(sub=sky_sims_id)
            param_res_compsep = np.load(fname_compsepresults, allow_pickle=True)["x"]
            params_names = np.load(fname_compsepresults, allow_pickle=True)["params_names"]
            param_res_list.append(param_res_compsep)
            last_valid_id = sky_sims_id
            print('params_names', params_names)
            if config.parametric_sep_pars.megabuster_options.use_hessienne:
                cov.append(np.load(fname_compsepresults, allow_pickle=True)["Cov"])
            if config.parametric_sep_pars.megabuster_options.use_hmc:
                mc_samples.append(np.load(fname_compsepresults, allow_pickle=True)["mc_samples"].item())
        except FileNotFoundError:
            logger.warning(
                f"Component separation results for simulation {sky_sims_id} not found, skipping."
            )
            continue
    param_res_list = np.array(param_res_list)

    # Plotting the statistics of the component separation results
    plot_dir = manager.path_to_components_plots
    res_compsep_last = np.load(
        manager.get_path_to_compsep_results(sub=last_valid_id), allow_pickle=True
    )

    # plotting histograms of result parameters
    fig, axes = plt.subplots(1, res_compsep_last["params"].shape[0], figsize=(20, 9))
    axes = np.atleast_1d(axes)
    for i, (ax, param_name) in enumerate(zip(axes, res_compsep_last["params"], strict=False)):
        ax.hist(param_res_list[:, i], density=True)

        ax.set_xlabel(param_name)
        # title showing mean and std:
        mean_param = np.mean(param_res_list[:, i])
        std_param = np.std(param_res_list[:, i])
        title = f"{param_name}\n{mean_param:.8g} ± {std_param:.8g}"
        # adding i*"\n" allows to hack around overlapping titles...
        ax.set_title(title, fontsize=9, wrap=True)

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig(plot_dir / Path("statistics_compsep.png"))  # , bbox_inches='tight')
    plt.close()

    #Plot compsep corner plot using Hessienne of HMC according to config file
    if config.parametric_sep_pars.megabuster_options.use_hessienne:
        corner_plot(manager, config, cov, param_res_list, params_names)
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