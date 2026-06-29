import argparse
from pathlib import Path

from matplotlib import cm

from megatop import Config, DataManager
from megatop.utils import Timer, logger
from megatop.utils.mask import (
    get_spin_derivatives,
)
from megatop.utils.plot import single_map_plotter


def plotter(manager: DataManager, config: Config):
    cmap = cm.RdBu
    cmap.set_under("w")
    plot_dir = manager.path_to_masks_plots
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Plotting hits map
    nhits = config.landscape.read_map(manager.path_to_common_nhits_map)
    single_map_plotter(
        nhits, config, plot_dir / "nhits_map.png", title="Common hits map", cmap=cmap
    )

    # Plotting binary mask map
    binary_mask = config.landscape.read_map(manager.path_to_binary_mask)
    single_map_plotter(
        binary_mask,
        config,
        plot_dir / "binary_mask.png",
        title="Binary mask, derived from hits map",
        cmap=cmap,
    )

    # Plotting galactic mask
    if config.masks_pars.include_galactic:
        galactic_mask = config.landscape.read_map(manager.path_to_galactic_mask)
        single_map_plotter(
            galactic_mask,
            config,
            plot_dir / f"{config.masks_pars.galactic_mask_name}_{config.masks_pars.gal_key}.png",
            title=f"Galactic map {config.masks_pars.gal_key}",
            cmap=cmap,
        )

    # Plotting point source mask
    if config.masks_pars.include_sources:
        point_source_mask = config.landscape.read_map(manager.path_to_sources_mask)
        single_map_plotter(
            point_source_mask,
            config,
            plot_dir / "point_source_mask.png",
            title="Point source mask",
            cmap=cmap,
        )

    # Plotting final analysis mask
    final_mask = config.landscape.read_map(manager.path_to_analysis_mask)
    single_map_plotter(
        final_mask, config, plot_dir / "analysis_mask.png", title="Final analysis mask", cmap=cmap
    )

    # CAR needs an explicit band limit; HEALPix keeps its nside-derived default
    first, second = get_spin_derivatives(final_mask, lmax=config.lmax if config.is_car else None)
    single_map_plotter(
        first,
        config,
        plot_dir / "analysis_mask_first.png",
        title="First spin derivative of the final analysis mask",
        cmap=cmap,
    )
    single_map_plotter(
        second,
        config,
        plot_dir / "analysis_mask_second.png",
        title="Second spin derivative of the final analysis mask",
        cmap=cmap,
    )

    # if config.masks_pars.DEBUG_output_apod_binary_mask:
    #     apod_binary_mask = hp.read_map(manager.path_to_apod_binary_mask)

    #     plt.figure(figsize=(16, 9))
    #     hp.mollview(
    #         apod_binary_mask,
    #         cmap=cmap,
    #         cbar=True,
    #         title="Apodized binary mask (no nhits rescaling)",
    #     )
    #     hp.graticule()
    #     plt.savefig(plot_dir / "apodized_binary_mask.png")
    #     plt.clf()


def main():
    parser = argparse.ArgumentParser(description="Plotter for mask_hanlder output")
    parser.add_argument("--config", type=Path, help="config file")
    args = parser.parse_args()
    if args.config is None:
        logger.warning("No config file provided, using example config")
        config = Config.get_example()
    else:
        config = Config.load_yaml(args.config)
    manager = DataManager(config)
    manager.dump_config()

    logger.info("Plotting mask outputs...")
    with Timer("mask-plotter"):
        plotter(manager, config)


if __name__ == "__main__":
    main()
