import argparse
from pathlib import Path

import healpy as hp
import numpy as np

from megatop import DataManager
from megatop.config import Config
from megatop.utils import Timer, logger
from megatop.utils.plot import freq_maps_plotter, plotTTEEBB
from megatop.utils.preproc import _apply_binary_mask


def plot_noisecov(manager, config, maps=True, cls=True):
    plot_dir = manager.path_to_covar_plots
    plot_dir.mkdir(parents=True, exist_ok=True)

    fname_noise_cov_maps = manager.path_to_pixel_noisecov
    noise_cov_maps = np.load(fname_noise_cov_maps)

    noise_cov_maps = _apply_binary_mask(manager, noise_cov_maps, unseen=True)

    if maps:
        freq_maps_plotter(config, noise_cov_maps, plot_dir, "noise_cov_maps")

    if cls:
        lmax = 3 * config.nside
        spectra_array = []
        for i in range(len(config.frequencies)):
            spectra_array.append(hp.anafast(noise_cov_maps[i], lmax=lmax))
        spectra_array = np.array(spectra_array)

        plotTTEEBB(
            plot_dir=plot_dir,
            freqs=config.frequencies,
            Cl=spectra_array,
            save_name="spectra_noise_cov_anafast",
            y_axis_label=r"$D_\ell$ noise covariance",
            use_D_ell=True,
            lims_x=None,
            lims_y=None,
        )


def main():
    parser = argparse.ArgumentParser(description="Plotter for pixel noise cov output")
    parser.add_argument("--config", type=Path, help="config file")
    args = parser.parse_args()
    if args.config is None:
        logger.warning("No config file provided, using example config")
        config = Config.get_example()
    else:
        config = Config.from_yaml(args.config)
    manager = DataManager(config)
    manager.dump_config()

    logger.info("Plotting noise cov outputs...")
    timer = Timer()
    timer.start("noisecov_plotter")

    plot_noisecov(manager, config)

    timer.stop("noisecov_plotter", "Plotting noise cov outputs")


if __name__ == "__main__":
    main()
