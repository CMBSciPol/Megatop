import argparse
from pathlib import Path

import numpy as np
from getdist import MCSamples, plots
from matplotlib import pyplot as plt

from megatop import Config, DataManager
from megatop.utils import logger


def plot_single_cornerplot(manager: DataManager, config: Config, id_sim: int | None = None):
    # Load parameters and mcmc chains:
    r_sim = config.map_sim_pars.r_input
    A_lens_sim = config.map_sim_pars.A_lens

    fname_chains = manager.get_path_to_mcmc_chains(sub=id_sim)
    mcmc = np.load(fname_chains, allow_pickle=True)
    chains = mcmc["mcmc_chains"]
    param_names = mcmc["param_names"]

    # Make plot:
    samples = MCSamples(samples=chains, names=param_names, labels=param_names)
    print(samples.getMeans())

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
        logger.info(f"Plotting only sky simulation #{id_sim}")
    plot_single_cornerplot(manager, config, id_sim=id_sim)


if __name__ == "__main__":
    main()
