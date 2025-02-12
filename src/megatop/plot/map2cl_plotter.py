import argparse
from pathlib import Path

import numpy as np

from megatop import Config, DataManager
from megatop.utils import Timer, logger
from megatop.utils.plot import plot_all_Cls


def plot_map2cl(manager):
    plot_dir = manager.path_to_spectra_plots
    plot_dir.mkdir(parents=True, exist_ok=True)
    binning_info = np.load(manager.path_to_binning, allow_pickle=True)

    bin_centre_lminlmax = binning_info["bin_centre_lminlmax"]

    all_Cls = np.load(manager.path_to_cross_components_spectra, allow_pickle=True)
    plot_all_Cls(
        all_Cls,
        bin_centre_lminlmax,
        plot_dir,
        "component_spectra",
        use_D_ell=False,
        y_axis_label=r"$C_{\ell}$",
    )


def main():
    parser = argparse.ArgumentParser(description="Plotter for map2cl output")
    parser.add_argument("--config", type=Path, help="config file")
    args = parser.parse_args()
    if args.config is None:
        logger.warning("No config file provided, using example config")
        config = Config.get_example()
    else:
        config = Config.from_yaml(args.config)
    manager = DataManager(config)
    manager.dump_config()

    logger.info("Plotting map2cl outputs...")
    timer = Timer()
    timer.start("map2cl_plotter")

    plot_map2cl(manager)

    timer.stop("map2cl_plotter")


if __name__ == "__main__":
    main()
