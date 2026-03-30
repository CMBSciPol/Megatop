import argparse
from pathlib import Path

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np

from getdist import MCSamples, plots
from megatop import Config, DataManager
from megatop.utils import Timer, logger
from megatop.utils.mask import apply_binary_mask
from megatop.utils.plot import freq_maps_plotter


def plot_compsep(manager: DataManager, config: Config, id_sim: int | None = None):
    plot_dir = manager.path_to_components_plots
    plot_dir.mkdir(parents=True, exist_ok=True)

    fname_compmaps = manager.get_path_to_components_maps(sub=id_sim)
    comp_maps = np.load(fname_compmaps)

    binary_mask = hp.read_map(manager.path_to_binary_mask)
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


def plot_compsep_stats(manager: DataManager, config: Config):
    if config.map_sim_pars.n_sim == 0 or config.map_sim_pars.n_sim is None:
        logger.info("No sky simulations, skipping component separation statistics plotter.")
        return

    param_res_list = []
    for sky_sims_id in range(config.map_sim_pars.n_sim):
        fname_compsepresults = manager.get_path_to_compsep_results(sub=sky_sims_id)
        param_res_compsep = np.load(fname_compsepresults, allow_pickle=True)["x"]
        param_res_list.append(param_res_compsep)
    param_res_list = np.array(param_res_list)

    # Plotting the statistics of the component separation results
    plot_dir = manager.path_to_components_plots
    res_compsep_last = np.load(fname_compsepresults, allow_pickle=True)

    # plotting histograms of result parameters
    fig, axes = plt.subplots(1, res_compsep_last["params"].shape[0], figsize=(12*2, 5))
    axes = np.atleast_1d(axes)
    for i, (ax, param_name) in enumerate(zip(axes, res_compsep_last["params"], strict=False)):
        ax.hist(param_res_list[:, i], density=True)

        ax.set_xlabel(param_name)
        # title showing mean and std:
        mean_param = np.mean(param_res_list[:, i])
        std_param = np.std(param_res_list[:, i])
        title = f"{param_name} = {mean_param:.5} +/- {std_param:.5}" + i * "\n"
        # adding i*"\n" allows to hack around overlapping titles...
        ax.set_title(title)

    plt.savefig(plot_dir / Path("statistics_compsep.png"))  # , bbox_inches='tight')
    plt.close()

    return

def plot_single_cornerplot(manager: DataManager, config: Config, id_sim: int | None = None):
    """Plot corner plot resulting from the MCMC for a single sky realization."""
    # Load parameters and mcmc chains:
    r_sim = config.map_sim_pars.r_input
    A_lens_sim = config.map_sim_pars.A_lens
    Birefringence = config.map_sim_pars.Birefringence

    fname_chains = manager.get_path_to_mcmc_chains_compsep(sub=id_sim)
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
        #markers={"r": r_sim, "A_{lens}": A_lens_sim,"Birefringence": Birefringence},
        
    )

    # Save figure:
    plot_dir = manager.path_to_mcmc_plots
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_name = f"corner_plot_skysim{id_sim}"
    plt.savefig(plot_dir / plot_name, bbox_inches="tight")
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
        logger.info("Plotting only simulation #0")
        id_sim = 0

    logger.info("Plotting comp sep outputs...")
    with Timer("comp-sep-plotter"):
        plot_compsep(manager, config, id_sim=id_sim)
        plot_compsep_stats(manager, config)


if __name__ == "__main__":
    main()
