import argparse
from pathlib import Path

import numpy as np

from megatop import Config, DataManager
from megatop.utils import Timer, logger
from megatop.utils.plot import plot_all_Cls


def plot_map2cl(manager, id_sim=None):
    plot_dir = manager.path_to_spectra_plots
    plot_dir.mkdir(parents=True, exist_ok=True)

    bin_centre_lminlmax = np.load(manager.path_to_binning, allow_pickle=True)["bin_centre_lminlmax"]

    path_all_Cls = manager.get_path_to_spectra_cross_components(sub=id_sim)
    all_Cls = np.load(path_all_Cls, allow_pickle=True)
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
        config = Config.load_yaml(args.config)
    manager = DataManager(config)
    manager.dump_config()

    logger.info("Plotting map2cl outputs...")
    timer = Timer()
    timer.start("map2cl_plotter")

    n_sim_sky = config.map_sim_pars.n_sim
    if n_sim_sky == 0:
        id_sim = None
    else:
        logger.info("Plotting only simulation #0")
        id_sim = 0

    plot_map2cl(manager, id_sim=id_sim)

    timer.stop("map2cl_plotter")


if __name__ == "__main__":
    main()
