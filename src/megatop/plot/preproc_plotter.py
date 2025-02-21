import argparse
from pathlib import Path

import healpy as hp
import numpy as np

from megatop import Config, DataManager
from megatop.utils import Timer, logger
from megatop.utils.mask import apply_binary_mask
from megatop.utils.plot import freq_maps_plotter, plotTTEEBB


def plot_preprocessed_maps(manager, config, maps=True, cls=True):
    plot_dir = manager.path_to_preproc_plots
    plot_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Plotting pre-processing outputs")

    with Timer("load-freq-maps"):
        preproc_maps_fname = manager.get_path_to_preprocessed_maps()
        logger.debug(f"Loading input maps from {preproc_maps_fname}")
        freq_maps_preprocessed = np.load(preproc_maps_fname)
        binary_mask = hp.read_map(manager.path_to_binary_mask)

        freq_maps_preprocessed = apply_binary_mask(
            freq_maps_preprocessed, binary_mask=binary_mask, unseen=True
        )

    if maps:  # Plotting the maps
        freq_maps_plotter(config, freq_maps_preprocessed, plot_dir, "pre_processed_maps")

    if cls:  # plotting the spectra
        lmax = 3 * config.nside
        spectra_array = []
        for i in range(len(config.frequencies)):
            spectra_array.append(hp.anafast(freq_maps_preprocessed[i], lmax=lmax))
        spectra_array = np.array(spectra_array)

        plotTTEEBB(
            plot_dir=plot_dir,
            freqs=config.frequencies,
            Cl=spectra_array,
            save_name="spectra_pre_processed_anafast",
            y_axis_label=r"$C_\ell$ pre-processed",
            use_D_ell=False,
            lims_x=None,
            lims_y=None,
        )


def main():
    parser = argparse.ArgumentParser(description="Plotter for preprocessing output")
    parser.add_argument("--config", type=Path, help="config file")
    args = parser.parse_args()
    if args.config is None:
        logger.warning("No config file provided, using example config")
        config = Config.get_example()
    else:
        config = Config.load_yaml(args.config)
    manager = DataManager(config)
    manager.dump_config()

    logger.info("Plotting preprocessing outputs...")

    with Timer("preproc-plotter"):
        plot_preprocessed_maps(manager, config)


if __name__ == "__main__":
    main()
