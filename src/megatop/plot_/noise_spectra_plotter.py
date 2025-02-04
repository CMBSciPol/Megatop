import argparse
from pathlib import Path

import numpy as np
import pymaster as nmt

from megatop import DataManager
from megatop.config import Config
from megatop.utils import Timer, logger
from megatop.utils.mock import _get_Cl_CMB_model_from_manager
from megatop.utils.plot import plot_all_Cls, plot_all_Cls_diff


def plot_noise_spectra(manager, config):
    plot_dir = manager.path_to_spectra_plots
    plot_dir.mkdir(parents=True, exist_ok=True)
    binning_info = np.load(manager.path_to_binning, allow_pickle=True)

    bin_centre_lminlmax = binning_info["bin_centre_lminlmax"]

    all_noise_Cls = np.load(manager.path_to_noise_cross_components_spectra, allow_pickle=True)
    plot_all_Cls(
        all_noise_Cls,
        bin_centre_lminlmax,
        plot_dir,
        "noise_post_compsep_spectra",
        use_D_ell=True,
        y_axis_label=r"$D_{\ell}$",
    )

    all_Cls = np.load(manager.path_to_cross_components_spectra, allow_pickle=True)

    debiased_cls = {}
    for key_cls, key_noise_cls in zip(all_Cls.keys(), all_noise_Cls.keys(), strict=False):
        debiased_cls[key_cls] = all_Cls[key_cls] - all_noise_Cls[key_noise_cls]

    plot_all_Cls(
        debiased_cls,
        bin_centre_lminlmax,
        plot_dir,
        "debiased_post_compsep_spectra",
        use_D_ell=True,
        y_axis_label=r"$D_{\ell}$",
    )

    Cl_cmb_model = _get_Cl_CMB_model_from_manager(manager)
    nmt_bins = nmt.NmtBin.from_edges(binning_info["bin_low"], binning_info["bin_high"] + 1)

    bined_Cl_cmb_model = nmt_bins.bin_cell(Cl_cmb_model[0, :, : 3 * config.nside])[
        :, binning_info["bin_index_lminlmax"]
    ]

    bined_Cl_cmb_model_dict = {
        "CMBxCMB": [
            bined_Cl_cmb_model[1],
            bined_Cl_cmb_model[4],
            bined_Cl_cmb_model[4],
            bined_Cl_cmb_model[2],
        ]
    }
    diabiased_cls_CMB_only = {"CMBxCMB": debiased_cls["CMBxCMB"]}

    plot_all_Cls_diff(
        diabiased_cls_CMB_only,
        bin_centre_lminlmax,
        bined_Cl_cmb_model_dict,
        plot_dir,
        "diff_debiased_CMB_spectra_vs_model",
        use_D_ell=True,
        y_axis_label=r"$D_{\ell}$",
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
    timer.start("Noise_spectra_plotter")

    plot_noise_spectra(manager, config)

    timer.stop("Noise_spectra_plotter")


if __name__ == "__main__":
    main()
