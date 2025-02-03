import argparse
from pathlib import Path

import healpy as hp
import numpy as np

from megatop import DataManager
from megatop.config import Config
from megatop.utils import Timer, logger
from megatop.utils.plot import freq_maps_plotter, plotTTEEBB
from megatop.utils.preproc import _apply_binary_mask


def plot_preprocessed_maps(manager, config, maps=True, cls=True):
    timer = Timer()
    plot_dir = manager.path_to_preproc_plots
    plot_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Plotting pre-processing outputs")

    timer.start("loading_maps")
    preproc_maps_fname = manager.get_path_to_preprocessed_maps()
    logger.debug(f"Loading input maps from {preproc_maps_fname}")
    freq_maps_preprocessed = np.load(preproc_maps_fname)
    timer.stop("loading_maps", "Loading pre-processed frequency maps")
    freq_maps_preprocessed = _apply_binary_mask(manager, freq_maps_preprocessed, unseen=True)

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
            y_axis_label=r"$D_\ell$ pre-processed",
            use_D_ell=True,
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
        config = Config.from_yaml(args.config)
    manager = DataManager(config)
    manager.dump_config()

    logger.info("Plotting preprocessing outputs...")
    timer = Timer()
    timer.start("preproc_plotter")

    plot_preprocessed_maps(manager, config)

    timer.stop("preproc_plotter", "Plotting preprocessing outputs")


if __name__ == "__main__":
    main()
