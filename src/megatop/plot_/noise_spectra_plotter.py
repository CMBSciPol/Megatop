import argparse
from pathlib import Path

import numpy as np
import pymaster as nmt

from megatop import DataManager
from megatop.config import Config
from megatop.utils import Timer, logger
from megatop.utils.mock import get_Cl_CMB_model_from_manager
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
        use_D_ell=False,
        y_axis_label=r"$C_{\ell}$",
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
        use_D_ell=False,
        y_axis_label=r"$C_{\ell}$",
    )

    Cl_cmb_model = get_Cl_CMB_model_from_manager(manager)[0, :, : 3 * config.nside]
    nmt_bins = nmt.NmtBin.from_edges(binning_info["bin_low"], binning_info["bin_high"] + 1)

    bined_Cl_cmb_model = nmt_bins.bin_cell(Cl_cmb_model)[:, binning_info["bin_index_lminlmax"]]

    bined_Cl_cmb_model_dict = {
        "CMBxCMB": [
            bined_Cl_cmb_model[1],
            bined_Cl_cmb_model[4],
            bined_Cl_cmb_model[4],
            bined_Cl_cmb_model[2],
        ]
    }
    diabiased_cls_CMB_only = {"CMBxCMB": debiased_cls["CMBxCMB"]}
    cls_CMB_only = {"CMBxCMB": all_Cls["CMBxCMB"]}

    bined_biased_cl_cmb_model = {
        "CMBxCMB": bined_Cl_cmb_model_dict["CMBxCMB"] + all_noise_Cls["Noise_CMBxNoise_CMB"]
    }

    # effective_beam_CMB = get_common_beam_wpix(config.pre_proc_pars.common_beam_correction, config.nside)
    # bined_effective_beam_CMB = nmt_bins.bin_cell(effective_beam_CMB[:-1])[binning_info["bin_index_lminlmax"]]
    # import healpy as hp
    # wpix_out = hp.pixwin(config.nside, pol=True, lmax=3 * config.nside)[1]  # Pixel window function of output maps
    # bined_wpix = nmt_bins.bin_cell(wpix_out[:-1])[binning_info["bin_index_lminlmax"]]

    # Bl_gauss_common = hp.gauss_beam(
    # np.radians(config.pre_proc_pars.common_beam_correction / 60), lmax=3 * config.nside, pol=True
    # )[:, 1]
    # bined_common = nmt_bins.bin_cell(Bl_gauss_common[:-1])[binning_info["bin_index_lminlmax"]]

    # A_maxL = np.load(manager.path_to_compsep_results, allow_pickle=True)["A_maxL"]
    # # W might be more appropriate but it's last dimension is a map, which makes things ill defined
    # effective_beam_all = get_effective_beam_noise_preproc(config, A_maxL)[0, 0, :-1]
    # effective_beam_all /= np.max(effective_beam_all)
    # bined_effective_beam_all = nmt_bins.bin_cell(effective_beam_all)[binning_info["bin_index_lminlmax"]]

    plot_all_Cls_diff(
        diabiased_cls_CMB_only,
        bin_centre_lminlmax,
        bined_Cl_cmb_model_dict,
        plot_dir,
        "diff_debiased_CMB_spectra_vs_model",
        use_D_ell=False,
        y_axis_label=r"$C_{\ell}$",
        # beam_dict={'eff':bined_effective_beam_CMB, 'wpix':bined_wpix, 'common':bined_common, 'eff_all':bined_effective_beam_all}
    )

    plot_all_Cls_diff(
        cls_CMB_only,
        bin_centre_lminlmax,
        bined_biased_cl_cmb_model,
        plot_dir,
        "diff_CMB_spectra_vs_biased_model",
        use_D_ell=False,
        y_axis_label=r"$C_{\ell}$",
        # beam_dict={'eff':bined_effective_beam_CMB, 'wpix':bined_wpix, 'common':bined_common, 'eff_all':bined_effective_beam_all}
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

    logger.info("Plotting Noise spectra outputs...")
    timer = Timer()
    timer.start("Noise_spectra_plotter")

    plot_noise_spectra(manager, config)

    timer.stop("Noise_spectra_plotter")


if __name__ == "__main__":
    main()
