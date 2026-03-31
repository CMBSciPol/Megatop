import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from megatop import Config, DataManager
from megatop.utils import Timer, logger
from megatop.utils.plot import plot_all_Cls


def plot_map2cl(manager, id_sim=None):
    plot_dir = manager.path_to_spectra_plots
    plot_dir.mkdir(parents=True, exist_ok=True)

    bin_centre_lminlmax = np.load(manager.path_to_binning, allow_pickle=True)["bin_centre_lminlmax"]
    bin_index_lminlmax = np.load(manager.path_to_binning, allow_pickle=True)["bin_index_lminlmax"]
    path_all_Cls = manager.get_path_to_spectra_cross_components(sub=id_sim)
    all_Cls_ = np.load(path_all_Cls, allow_pickle=True)
    all_Cls_binranged = {}
    for k in all_Cls_:
        all_Cls_binranged[k] = all_Cls_[k][:, bin_index_lminlmax]
    plot_all_Cls(
        all_Cls_binranged,
        bin_centre_lminlmax,
        plot_dir,
        "component_spectra",
        use_D_ell=False,
        y_axis_label=r"$C_{\ell}$",
    )


def plot_all_cmb_spectra(manager, config):
    plot_dir = manager.path_to_spectra_plots
    plot_dir.mkdir(parents=True, exist_ok=True)

    binning_info = np.load(manager.path_to_binning, allow_pickle=True)
    bin_centre_lminlmax = binning_info["bin_centre_lminlmax"]
    bin_index_lminlmax = np.load(manager.path_to_binning, allow_pickle=True)["bin_index_lminlmax"]

    fig_EE, ax_EE = plt.subplots()
    fig_BB, ax_BB = plt.subplots()

    average_CMB = np.zeros([4, len(bin_centre_lminlmax)])
    num_loaded_id = 0
    for id_sim in range(config.map_sim_pars.n_sim):
        try:
            fname_Cls = manager.get_path_to_spectra_cross_components(sub=id_sim)
            all_Cls = np.load(fname_Cls, allow_pickle=True)
            all_Cls_CMB = all_Cls["CMBxCMB"][:, bin_index_lminlmax]

            average_CMB += all_Cls["CMBxCMB"][:, bin_index_lminlmax]

            ax_EE.plot(
                bin_centre_lminlmax,
                all_Cls_CMB[0],
                label="Estimated Noise_CMB EE" if id_sim == 0 else None,
                linestyle="-",
                color="darkblue",
                alpha=0.2,
            )
            ax_BB.plot(
                bin_centre_lminlmax,
                all_Cls_CMB[-1],
                label="Estimated Noise_CMB BB" if id_sim == 0 else None,
                linestyle="-",
                color="darkblue",
                alpha=0.2,
            )
            num_loaded_id += 1

        except FileNotFoundError:
            logger.warning(f"Spectra file not found for id_sim={id_sim} at Path:{fname_Cls}")

    # WARNING: here we average the noise Nl (already averaged over noise sims) over the different sky sims
    # average_CMB /= config.map_sim_pars.n_sim
    average_CMB /= num_loaded_id  # config.map_sim_pars.n_sim

    ax_EE.plot(
        bin_centre_lminlmax,
        average_CMB[0],
        label="Mean noise EE",
        color="brown",
        linestyle="-",
    )
    ax_BB.plot(
        bin_centre_lminlmax,
        average_CMB[-1],
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
    plot_all_cmb_spectra(manager, config)

    timer.stop("map2cl_plotter")


if __name__ == "__main__":
    main()
