import argparse
from pathlib import Path

import healpy as hp
import matplotlib.pyplot as plt
from matplotlib import cm

from megatop import Config, DataManager
from megatop.utils import Timer, logger
from megatop.utils.mask import (
    get_spin_derivatives,
)


def plotter(manager: DataManager, config: Config):
    cmap = cm.RdBu
    cmap.set_under("w")
    plot_dir = manager.path_to_masks_plots
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Plotting hits map
    nhits = hp.read_map(manager.path_to_nhits_map)
    plt.figure(figsize=(16, 9))
    hp.mollview(nhits, cmap=cmap, cbar=True, title="Hits map")
    hp.graticule()
    plt.savefig(plot_dir / "nhits_map.png")
    plt.clf()

    # Plotting binary mask map
    binary_mask = hp.read_map(manager.path_to_binary_mask)
    plt.figure(figsize=(16, 9))
    hp.mollview(binary_mask, cmap=cmap, cbar=True, title="Binary mask, derived from hits map")
    hp.graticule()
    plt.savefig(plot_dir / "binary_mask.png")
    plt.clf()

    # Plotting galactic mask
    if config.masks_pars.include_galactic:
        galactic_mask = hp.read_map(manager.path_to_galactic_mask)
        plt.figure(figsize=(16, 9))
        hp.mollview(
            galactic_mask,
            cmap=cmap,
            cbar=True,
            title=f"Galactic map {config.masks_pars.gal_key}",
        )
        hp.graticule()
        plt.savefig(
            plot_dir / f"{config.masks_pars.galactic_mask_name}_{config.masks_pars.gal_key}.png",
        )
        plt.clf()

    # Plotting point source mask
    if config.masks_pars.include_sources:
        point_source_mask = hp.read_map(manager.path_to_sources_mask)
        plt.figure(figsize=(16, 9))
        hp.mollview(point_source_mask, cmap=cmap, cbar=True, title="Point source mask")
        hp.graticule()
        plt.savefig(plot_dir / "point_source_mask.png")
        plt.clf()

    # Plotting final analysis mask
    final_mask = hp.read_map(manager.path_to_analysis_mask)
    plt.figure(figsize=(16, 9))
    hp.mollview(final_mask, cmap=cmap, cbar=True, title="Final analysis mask")
    hp.graticule()
    plt.savefig(plot_dir / "analysis_mask.png")
    plt.clf()

    first, second = get_spin_derivatives(final_mask)
    # Plot first spin derivative of analysis mask
    plt.figure(figsize=(16, 9))
    hp.mollview(
        first, title="First spin derivative of the final analysis mask", cmap=cmap, cbar=True
    )
    hp.graticule()
    plt.savefig(plot_dir / "analysis_mask_first.png")
    plt.clf()

    # Plot second spin derivative of analysis mask
    plt.figure(figsize=(16, 9))
    hp.mollview(
        second, title="Second spin derivative of the final analysis mask", cmap=cmap, cbar=True
    )
    hp.graticule()
    plt.savefig(plot_dir / "analysis_mask_second.png")
    plt.clf()


def main():
    parser = argparse.ArgumentParser(description="Plotter for mask_hanlder output")
    parser.add_argument("--config", type=Path, help="config file")
    args = parser.parse_args()
    if args.config is None:
        logger.warning("No config file provided, using example config")
        config = Config.get_example()
    else:
        config = Config.from_yaml(args.config)
    manager = DataManager(config)
    manager.dump_config()

    logger.info("Plotting mask outputs...")
    with Timer("mask-plotter"):
        plotter(manager, config)


if __name__ == "__main__":
    main()
