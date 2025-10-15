import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from megatop import Config, DataManager
from megatop.utils import Timer, logger
from megatop.utils.binning import load_nmt_binning
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


def plot_harmonic_byproducts(manager, id_sim=None):
    nmt_bins = load_nmt_binning(manager)
    ells = nmt_bins.get_effective_ells()

    path = manager.get_path_to_spectra(sub=id_sim)
    fname_Cl_WmaxL = path / Path("Cl_WmaxL.npy")
    fname_Cl_effective_TF = path / Path("Cl_effective_TF_normalized.npy")
    # fname_Cl_effective_TF_inv = path / Path("Cl_effective_TF_inv_normalized.npy")
    Cl_WmaxL = np.load(fname_Cl_WmaxL)
    normalized_Cl_effective_TF = np.load(fname_Cl_effective_TF)

    components_list = ["CMB", "Dust", "Synch"]
    for i in range(Cl_WmaxL.shape[0]):
        for j in range(i, Cl_WmaxL.shape[1]):
            fig_W, ax_W = plt.subplots(2, 2, figsize=(8, 6))
            for f in range(Cl_WmaxL.shape[2]):
                ax_W[0, 0].plot(ells, Cl_WmaxL[i, j, f, 0, :], color=f"C{f}")
                ax_W[0, 0].plot(ells, -Cl_WmaxL[i, j, f, 0, :], color=f"C{f}", linestyle="--")
                ax_W[0, 1].plot(ells, Cl_WmaxL[i, j, f, 1, :], label=f"Freq #{f}", color=f"C{f}")
                # ax_W[0,1].plot(ells, -Cl_WmaxL[i,j,f,1,:], color=f"C{f}", linestyle='--')
                ax_W[1, 0].plot(ells, Cl_WmaxL[i, j, f, 2, :], color=f"C{f}")
                # ax_W[1,0].plot(ells, -Cl_WmaxL[i,j,f,2,:], color=f"C{f}", linestyle='--')
                ax_W[1, 1].plot(ells, Cl_WmaxL[i, j, f, 3, :], color=f"C{f}")
                ax_W[1, 1].plot(ells, -Cl_WmaxL[i, j, f, 3, :], color=f"C{f}", linestyle="--")
            ax_W[0, 0].set_title(
                f"W_maxL EE for components {components_list[i]} and {components_list[j]}"
            )
            ax_W[0, 1].set_title("W_maxL EB")
            ax_W[1, 0].set_title("W_maxL BE")
            ax_W[1, 1].set_title("W_maxL BB")
            ax_W[0, 1].legend()

            ax_W[0, 0].set_yscale("log")
            # ax_W[0,1].set_yscale("log")
            # ax_W[1,0].set_yscale("log")
            ax_W[1, 1].set_yscale("log")
            plt.savefig(
                manager.path_to_spectra_plots
                / Path(f"WmaxL_comp_{components_list[i]}_{components_list[j]}.png")
            )
            plt.close(fig_W)

    spectra_list = ["EE", "EB", "BE", "BB"]

    transfer_freq = []
    for tf_path in manager.get_TF_filenames():
        transfer = np.load(tf_path, allow_pickle=True)["full_tf"]
        transfer_freq.append(transfer)
    transfer_freq = np.array(transfer_freq)
    transfer_freq_pol = transfer_freq[:, -4:, -4:]  # keeping only polarised components

    for i in range(normalized_Cl_effective_TF.shape[0]):
        for j in range(i, normalized_Cl_effective_TF.shape[1]):
            fig_TF, ax_TF = plt.subplots(4, 4, figsize=(8, 6))
            for k in range(normalized_Cl_effective_TF.shape[2]):
                for L in range(normalized_Cl_effective_TF.shape[3]):
                    for f in range(transfer_freq_pol.shape[0]):
                        ax_TF[k, L].plot(
                            ells,
                            transfer_freq_pol[f, k, L, :],
                            color=f"C{f}",
                            alpha=0.3,
                            label=f"Freq #{f}" if (k == 0 and L == 1) else None,
                        )
                    ax_TF[k, L].plot(
                        ells,
                        normalized_Cl_effective_TF[i, j, k, L, :],
                        label="Eff. TF",
                        color="black",
                    )
                    # if (k==0 and l==0) or (k==3 and l==3):
                    ax_TF[k, L].set_yscale("log")
                    ax_TF[k, L].axhline(1, color="k", linestyle="--", alpha=0.5)
                    if k == 0 and L == 0:
                        ax_TF[k, L].set_title(
                            f"Eff. TF {spectra_list[k]}-->{spectra_list[L]}\n{components_list[i]}x{components_list[j]}"
                        )
                    else:
                        ax_TF[k, L].set_title(f"{spectra_list[k]}-->{spectra_list[L]}")
            ax_TF[0, 1].legend()

            plt.tight_layout()
            plt.savefig(
                manager.path_to_spectra_plots
                / Path(f"Eff_TF_comp_{components_list[i]}_{components_list[j]}.png")
            )
            plt.close(fig_TF)
    # Effective TF only BB CMBxCMB:
    fig_TF_BB, ax_TF_BB = plt.subplots()
    fig_TF_BB_diff, ax_TF_BB_diff = plt.subplots()
    fig_TF_BB_relat_freq_diff, ax_TF_BB_relat_freq_diff = plt.subplots()
    fig_TF_BB_relat_eff_diff, ax_TF_BB_relat_eff_diff = plt.subplots()

    i = 0
    j = 0
    k = 3
    L = 3
    for f in range(transfer_freq_pol.shape[0]):
        ax_TF_BB.plot(
            ells, transfer_freq_pol[f, k, L, :], color=f"C{f}", alpha=0.3, label=f"Freq #{f}"
        )
        ax_TF_BB_diff.plot(
            ells,
            transfer_freq_pol[f, k, L, :] - normalized_Cl_effective_TF[i, j, k, L, :],
            color=f"C{f}",
            alpha=0.3,
            label=f"Freq #{f}",
        )
        ax_TF_BB_relat_freq_diff.plot(
            ells,
            (-transfer_freq_pol[f, k, L, :] + normalized_Cl_effective_TF[i, j, k, L, :])
            / transfer_freq_pol[f, k, L, :]
            * 100,
            color=f"C{f}",
            alpha=0.3,
            label=f"Freq #{f}",
        )
        ax_TF_BB_relat_eff_diff.plot(
            ells,
            (transfer_freq_pol[f, k, L, :] - normalized_Cl_effective_TF[i, j, k, L, :])
            / normalized_Cl_effective_TF[i, j, k, L, :]
            * 100,
            color=f"C{f}",
            alpha=0.3,
            label=f"Freq #{f}",
        )

    ax_TF_BB.plot(ells, normalized_Cl_effective_TF[i, j, k, L, :], label="Eff. TF", color="black")
    ax_TF_BB.set_yscale("log")
    ax_TF_BB.axhline(1, color="k", linestyle="--", alpha=0.5)
    ax_TF_BB.set_title(f"Eff. TF BB\n{components_list[i]}x{components_list[j]}")
    ax_TF_BB.set_xlabel(r"$\ell$")
    ax_TF_BB.set_ylabel("Transfer function")
    ax_TF_BB.legend()

    ax_TF_BB_diff.axhline(0, color="k", linestyle="--", alpha=0.5)
    ax_TF_BB_diff.set_title(
        f"Eff. TF BB difference to individual freq.\n{components_list[i]}x{components_list[j]}"
    )
    ax_TF_BB_diff.set_xlabel(r"$\ell$")
    ax_TF_BB_diff.set_ylabel("relative difference Transfer function")
    ax_TF_BB_diff.legend()

    ax_TF_BB_relat_freq_diff.axhline(0, color="k", linestyle="--", alpha=0.5)
    ax_TF_BB_relat_freq_diff.set_title(
        f"Relat diff wrt freq TF Eff. TF BB to individual freq.\n{components_list[i]}x{components_list[j]}"
    )
    ax_TF_BB_relat_freq_diff.set_xlabel(r"$\ell$")
    ax_TF_BB_relat_freq_diff.set_ylabel("relative difference Transfer function / freq TF")
    ax_TF_BB_relat_freq_diff.legend()

    ax_TF_BB_relat_eff_diff.axhline(0, color="k", linestyle="--", alpha=0.5)
    ax_TF_BB_relat_eff_diff.set_title(
        f"Relat diff wrt effTF Eff. TF BB difference to individual freq.\n{components_list[i]}x{components_list[j]}"
    )
    ax_TF_BB_relat_eff_diff.set_xlabel(r"$\ell$")
    ax_TF_BB_relat_eff_diff.set_ylabel("relative difference Transfer function / Eff. TF")
    ax_TF_BB_relat_eff_diff.legend()

    fig_TF_BB.tight_layout()
    fig_TF_BB.savefig(manager.path_to_spectra_plots / Path("Eff_TF_BB_CMBxCMB.png"))

    fig_TF_BB_diff.tight_layout()
    fig_TF_BB_diff.savefig(manager.path_to_spectra_plots / Path("Eff_TF_BB_CMBxCMB_diff.png"))

    fig_TF_BB_relat_freq_diff.tight_layout()
    fig_TF_BB_relat_freq_diff.savefig(
        manager.path_to_spectra_plots / Path("Eff_TF_BB_CMBxCMB_relat_freq_diff.png")
    )

    fig_TF_BB_relat_eff_diff.tight_layout()
    fig_TF_BB_relat_eff_diff.savefig(
        manager.path_to_spectra_plots / Path("Eff_TF_BB_CMBxCMB_relat_eff_diff.png")
    )
    plt.close(fig_TF_BB)
    plt.close(fig_TF_BB_diff)
    plt.close(fig_TF_BB_relat_freq_diff)
    plt.close(fig_TF_BB_relat_eff_diff)

    # inverse_normalized_Cl_effective_TF = np.load(fname_Cl_effective_TF_inv)


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

    if config.pre_proc_pars.correct_for_TF and config.parametric_sep_pars.use_harmonic_compsep:
        logger.info("Also plotting W_Cl and Effective Transfer Function outputs...")
        plot_harmonic_byproducts(manager, id_sim=id_sim)

    timer.stop("map2cl_plotter")


if __name__ == "__main__":
    main()
