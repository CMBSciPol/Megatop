import argparse
from pathlib import Path

import numpy as np

from megatop import DataManager
from megatop.config import Config
from megatop.utils import Timer, logger
from megatop.utils.plot import freq_maps_plotter
from megatop.utils.preproc import _apply_binary_mask


def plot_compsep(manager, config):
    plot_dir = manager.path_to_components_plots
    plot_dir.mkdir(parents=True, exist_ok=True)

    comp_maps = np.load(manager.path_to_components_maps)
    comp_maps = _apply_binary_mask(manager, comp_maps, unseen=True)

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
    freq_maps_plotter(
        config,
        np.array([comp_maps[2]]),
        plot_dir,
        "synch_post_compsep_maps",
        component="Synch post-compsep",
    )


def main():
    parser = argparse.ArgumentParser(description="Plotter for component separation output")
    parser.add_argument("--config", type=Path, help="config file")
    args = parser.parse_args()
    if args.config is None:
        logger.warning("No config file provided, using example config")
        config = Config.get_example()
    else:
        config = Config.from_yaml(args.config)
    manager = DataManager(config)
    manager.dump_config()

    logger.info("Plotting comp sep outputs...")
    with Timer("comp-sep-plotter"):
        plot_compsep(manager, config)


if __name__ == "__main__":
    main()
