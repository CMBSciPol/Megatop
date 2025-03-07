import argparse
from pathlib import Path

import healpy as hp
import numpy as np

from megatop import Config, DataManager
from megatop.utils import Timer, logger
from megatop.utils.mask import apply_binary_mask
from megatop.utils.plot import freq_maps_plotter


def plot_compsep(manager, config, id_sim=None):
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


if __name__ == "__main__":
    main()
