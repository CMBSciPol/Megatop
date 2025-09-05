import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from megatop import Config, DataManager
from megatop.utils import Timer, logger
from megatop.utils.binning import load_nmt_binning
from megatop.utils.mock import get_Cl_CMB_model_from_manager
from megatop.utils.plot import plot_all_Cls, plot_all_Cls_diff


def plot_all_noise_spectra(manager, config):
    plot_dir = manager.path_to_spectra_plots
    plot_dir.mkdir(parents=True, exist_ok=True)

    binning_info = np.load(manager.path_to_binning, allow_pickle=True)
    bin_centre_lminlmax = binning_info["bin_centre_lminlmax"]

    fig_EE, ax_EE = plt.subplots()
    fig_BB, ax_BB = plt.subplots()

    average_noise_CMB = np.zeros([4, len(bin_centre_lminlmax)])
    for id_sim in range(config.map_sim_pars.n_sim):
        fname_noise_Cls = manager.get_path_to_noise_spectra_cross_components(sub=id_sim)
        all_noise_Cls = np.load(fname_noise_Cls, allow_pickle=True)
        all_noise_Cls_CMB = all_noise_Cls["Noise_CMBxNoise_CMB"]

        average_noise_CMB += all_noise_Cls["Noise_CMBxNoise_CMB"]

        ax_EE.plot(
            bin_centre_lminlmax,
            all_noise_Cls_CMB[0],
            label="Estimated Noise_CMB EE" if id_sim == 0 else None,
            linestyle="-",
            color="darkblue",
            alpha=0.2,
        )
        ax_BB.plot(
            bin_centre_lminlmax,
            all_noise_Cls_CMB[-1],
            label="Estimated Noise_CMB BB" if id_sim == 0 else None,
            linestyle="-",
            color="darkblue",
            alpha=0.2,
        )
    average_noise_CMB /= config.map_sim_pars.n_sim

    ax_EE.plot(
        bin_centre_lminlmax,
        average_noise_CMB[0],
        label="Mean noise EE",
        color="brown",
        linestyle="-",
    )
    ax_BB.plot(
        bin_centre_lminlmax,
        average_noise_CMB[-1],
        label="Mean noise BB",
        color="brown",
        linestyle="-",
    )

    ax_EE.set_xlabel(r"$\ell$")
    ax_EE.set_ylabel(r"$C_{\ell}^{EE}$")
    ax_EE.legend()
    ax_EE.loglog()
    ax_EE.set_title("Noise_CMB EE spectra")
    fig_EE.savefig(plot_dir / "allskysims_NoiseCMB_EE_spectra.png")

    ax_BB.set_xlabel(r"$\ell$")
    ax_BB.set_ylabel(r"$C_{\ell}^{BB}$")
    ax_BB.legend()
    ax_BB.loglog()
    ax_BB.set_title("Noise_CMB BB spectra")
    fig_BB.savefig(plot_dir / "allskysims_NoiseCMB_BB_spectra.png")
    # closing figures
    plt.close(fig_EE)
    plt.close(fig_BB)


def plot_all_spectra(manager, config):
    plot_dir = manager.path_to_spectra_plots
    plot_dir.mkdir(parents=True, exist_ok=True)

    binning_info = np.load(manager.path_to_binning, allow_pickle=True)
    bin_centre_lminlmax = binning_info["bin_centre_lminlmax"]
    bin_index_lminlmax = binning_info["bin_index_lminlmax"]

    Cl_cmb_model = get_Cl_CMB_model_from_manager(manager)[0, :, : 3 * config.nside]
    nmt_bins = load_nmt_binning(manager)

    bined_Cl_cmb_model = nmt_bins.bin_cell(Cl_cmb_model)[:, bin_index_lminlmax]

    fig_EE, ax_EE = plt.subplots()
    fig_BB, ax_BB = plt.subplots()

    average_noise_CMB = np.zeros([4, len(bin_centre_lminlmax)])
    for id_sim in range(config.map_sim_pars.n_sim):
        fname_noise_Cls = manager.get_path_to_noise_spectra_cross_components(sub=id_sim)
        all_noise_Cls = np.load(fname_noise_Cls, allow_pickle=True)

        fname_all_Cls = manager.get_path_to_spectra_cross_components(sub=id_sim)
        all_Cls = np.load(fname_all_Cls, allow_pickle=True)

        debiased_cmb_cls = all_Cls["CMBxCMB"]
        average_noise_CMB += all_noise_Cls["Noise_CMBxNoise_CMB"]

        ax_EE.plot(
            bin_centre_lminlmax,
            debiased_cmb_cls[0],
            label="Estimated CMB EE (noisy)" if id_sim == 0 else None,
            linestyle="-",
            color="darkblue",
            alpha=0.2,  # if not negative_bins_in_EE else 1.0,
        )
        ax_BB.plot(
            bin_centre_lminlmax,
            debiased_cmb_cls[-1],
            label="Estimated CMB BB (noisy)" if id_sim == 0 else None,
            linestyle="-",
            color="darkblue",
            alpha=0.2,  # if not negative_bins_in_BB else 1.0,
        )

        negative_bins = debiased_cmb_cls[-1] < 0
        ax_BB.plot(
            bin_centre_lminlmax[negative_bins],
            np.abs(debiased_cmb_cls[-1][negative_bins]),
            label="ABS(Estimated CMB BB (noisy))" if id_sim == 0 else None,
            linestyle="--",
            color="green",
            alpha=0.2,
        )

    average_noise_CMB /= config.map_sim_pars.n_sim

    ax_EE.plot(
        bin_centre_lminlmax,
        bined_Cl_cmb_model[1] + average_noise_CMB[0],
        label="CMB EE model + mean noise",
        color="black",
        linestyle="--",
    )
    ax_BB.plot(
        bin_centre_lminlmax,
        bined_Cl_cmb_model[2] + average_noise_CMB[-1],
        label="CMB BB model + mean noise",
        color="black",
        linestyle="--",
    )

    ax_EE.plot(
        bin_centre_lminlmax,
        average_noise_CMB[0],
        label="Mean noise EE",
        color="brown",
        linestyle="-",
    )
    ax_BB.plot(
        bin_centre_lminlmax,
        average_noise_CMB[-1],
        label="Mean noise BB",
        color="brown",
        linestyle="-",
    )

    ax_EE.set_xlabel(r"$\ell$")
    ax_EE.set_ylabel(r"$C_{\ell}^{EE}$")
    ax_EE.legend()
    ax_EE.loglog()
    ax_EE.set_title("CMB EE spectra")
    fig_EE.savefig(plot_dir / "allskysims_CMB_EE_spectra.png")

    ax_BB.set_xlabel(r"$\ell$")
    ax_BB.set_ylabel(r"$C_{\ell}^{BB}$")
    ax_BB.legend()
    ax_BB.loglog()
    ax_BB.set_title("CMB BB spectra")
    fig_BB.savefig(plot_dir / "allskysims_CMB_BB_spectra.png")
    # closing figures
    plt.close(fig_EE)
    plt.close(fig_BB)


def plot_noise_spectra(manager, config, id_sim=None):
    plot_dir = manager.path_to_spectra_plots
    plot_dir.mkdir(parents=True, exist_ok=True)

    binning_info = np.load(manager.path_to_binning, allow_pickle=True)
    bin_centre_lminlmax = binning_info["bin_centre_lminlmax"]

    fname_noise_Cls = manager.get_path_to_noise_spectra_cross_components(sub=id_sim)
    all_noise_Cls = np.load(fname_noise_Cls, allow_pickle=True)

    plot_all_Cls(
        all_noise_Cls,
        bin_centre_lminlmax,
        plot_dir,
        "noise_post_compsep_spectra",
        use_D_ell=False,
        y_axis_label=r"$C_{\ell}$",
    )

    fname_all_Cls = manager.get_path_to_spectra_cross_components(sub=id_sim)
    all_Cls = np.load(fname_all_Cls, allow_pickle=True)

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
    nmt_bins = load_nmt_binning(manager)

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

    plot_all_Cls_diff(
        diabiased_cls_CMB_only,
        bin_centre_lminlmax,
        bined_Cl_cmb_model_dict,
        plot_dir,
        "diff_debiased_CMB_spectra_vs_model",
        use_D_ell=False,
        y_axis_label=r"$C_{\ell}$",
    )

    cls_CMB_only = {"CMBxCMB": all_Cls["CMBxCMB"]}
    bined_biased_cl_cmb_model = {
        "CMBxCMB": bined_Cl_cmb_model_dict["CMBxCMB"] + all_noise_Cls["Noise_CMBxNoise_CMB"]
    }

    plot_all_Cls_diff(
        cls_CMB_only,
        bin_centre_lminlmax,
        bined_biased_cl_cmb_model,
        plot_dir,
        "diff_CMB_spectra_vs_biased_model",
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

    logger.info("Plotting Noise spectra outputs...")
    timer = Timer()
    timer.start("Noise_spectra_plotter")

    n_sim_sky = config.map_sim_pars.n_sim
    if n_sim_sky == 0:
        id_sim = None
    else:
        logger.info("Plotting only simulation #0")
        id_sim = 0

    plot_noise_spectra(manager, config, id_sim=id_sim)

    if n_sim_sky != 0:
        logger.info("Plotting all spectra:")
        plot_all_spectra(manager, config)
        plot_all_noise_spectra(manager, config)

    timer.stop("Noise_spectra_plotter")


if __name__ == "__main__":
    main()
