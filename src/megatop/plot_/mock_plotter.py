import argparse
from pathlib import Path

import healpy as hp
import IPython
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

from megatop import DataManager
from megatop.config import Config
from megatop.utils import Timer, logger, mock
from megatop.utils.preproc import _apply_binary_mask, _read_input_maps


def plot_fiducial_spectra(manager):
    plot_dir = manager.path_to_mock_plots
    plot_dir.mkdir(parents=True, exist_ok=True)

    path_Cl_BB_lens = manager.path_to_lensed_scalar
    path_Cl_BB_prim_r1 = manager.path_to_unlensed_scalar_tensor_r1

    Cl_lens = hp.read_cl(path_Cl_BB_lens)
    Cl_prim = hp.read_cl(path_Cl_BB_prim_r1)[..., : Cl_lens.shape[-1]]
    Cl_BB_prim = manager._config.map_sim_pars.r_input * Cl_prim[2]
    Cl_BB_lens = manager._config.map_sim_pars.A_lens * Cl_lens[2]
    ell_range = np.arange(Cl_lens.shape[-1])
    todls = ell_range * (ell_range + 1) / 2.0 / np.pi

    plt.figure(figsize=(16, 9))

    plt.plot(ell_range, todls * Cl_prim[0], label="prim TT", color="C0")
    plt.plot(ell_range, todls * Cl_lens[0], label="lens TT", color="C0", ls="--")

    plt.plot(ell_range, todls * Cl_prim[1], label="prim EE", color="C1")
    plt.plot(ell_range, todls * Cl_lens[1], label="lens EE", color="C1", ls="--")

    plt.plot(ell_range, todls * Cl_BB_prim, label="prim BB", color="C2")
    plt.plot(ell_range, todls * Cl_BB_lens, label="lens BB", color="C2", ls="--")

    plt.plot(ell_range, todls * Cl_prim[3], label="prim TE", color="C3")
    plt.plot(ell_range, todls * Cl_lens[3], label="lens TE", color="C3", ls="--")
    plt.plot(ell_range, -todls * Cl_prim[3], label="prim TE", color="C3", alpha=0.5)
    plt.plot(ell_range, -todls * Cl_lens[3], label="lens TE", color="C3", ls="--", alpha=0.5)

    plt.loglog()
    plt.title("Fiducial power spectra")
    plt.xlabel(r"$\ell$")
    plt.ylabel(r"$D_\ell$")
    plt.xlim(2, 2000)
    plt.legend()
    plt.savefig(plot_dir / "fiducial_CMB_spectra.png", bbox_inches="tight")
    plt.clf()


def plotTTEEBB_diff(
    plot_dir,
    freqs,
    Cl_data,
    Cl_model,
    save_name,
    legend_labels=(r"label data $C_\ell$ $\nu=$", r"label model $C_\ell$ $\nu=$"),
    axis_labels=("y_axis_row0", "y_axis_row1"),
    use_D_ell=True,
    lims_x=(2, 2000),
    lims_y=(1e-2, 1e7),
):
    """
    This function plots the difference between the data and the model Cls. It directly saves the plot directly.

    Args:
        args: The parser arguments, containing the path to the global parameters file to set up metadata manager.
        Cl_data (ndarray): The data Cls, with shape (num_freq, num_spectra [TT,EE,BB], num_ell).
        Cl_model (ndarray): The model Cls, with shape (num_freq, num_spectra [TT,EE,BB], num_ell).
        save_name (str): The name of the file to save the plot. It will save the plot in the plots directory of the simulation output directory.
                         OR complete save path if you want to save it elsewhere.
        legend_labels (list): The labels for the legend of the plot.
        axis_labels (list): The labels for the x and y axes of the plot.

    Returns:
        None
    """

    ell = np.arange(0, Cl_data.shape[-1])
    norm = ell * (ell + 1) / 2 / np.pi

    if Cl_data.ndim == 2:
        Cl_data = Cl_data[np.newaxis, ...]
    if Cl_model.ndim == 2:
        Cl_model = Cl_model[np.newaxis, ...]
    Cl_model = Cl_model[..., ell]

    if not use_D_ell:
        norm = 1

    fig, ax = plt.subplots(2, 3, sharex=True, sharey="row", figsize=(15, 15))
    for f in range(Cl_data.shape[0]):
        ax[0][0].plot(ell, norm * Cl_data[f, 0], color="C" + str(f), ls="-", alpha=0.4)
        ax[0][1].plot(ell, norm * Cl_data[f, 1], color="C" + str(f), ls="-", alpha=0.4)
        ax[0][2].plot(
            ell,
            norm * Cl_data[f, 2],
            label=legend_labels[0] + str(freqs[f]) * (Cl_data.shape[0] != 1),
            color="C" + str(f),
            ls="-",
            alpha=0.4,
        )
        ax[0][0].plot(ell, norm * Cl_model[f, 0], color="C" + str(f), ls=":")
        ax[0][1].plot(ell, norm * Cl_model[f, 1], color="C" + str(f), ls=":")
        ax[0][2].plot(
            ell,
            norm * Cl_model[f, 2],
            label=legend_labels[1] + str(freqs[f]) * (Cl_data.shape[0] != 1),
            color="C" + str(f),
            ls=":",
        )

        zero_index_model0 = np.where(Cl_model[f, 0] != 0)[0]
        zero_index_model1 = np.where(Cl_model[f, 1] != 0)[0]
        zero_index_model2 = np.where(Cl_model[f, 2] != 0)[0]
        ax[1][0].plot(
            ell[zero_index_model0],
            (
                (Cl_data[f, 0] - Cl_model[f, 0])[zero_index_model0]
                / Cl_model[f, 0, zero_index_model0]
            ),
            color="C" + str(f),
            ls="-",
            alpha=0.4,
        )
        ax[1][1].plot(
            ell[zero_index_model1],
            (
                (Cl_data[f, 1] - Cl_model[f, 1])[zero_index_model1]
                / Cl_model[f, 1, zero_index_model1]
            ),
            color="C" + str(f),
            ls="-",
            alpha=0.4,
        )
        ax[1][2].plot(
            ell[zero_index_model2],
            (
                (Cl_data[f, 2] - Cl_model[f, 2])[zero_index_model2]
                / Cl_model[f, 2, zero_index_model2]
            ),
            color="C" + str(f),
            ls="-",
            alpha=0.4,
        )

    ax[0][0].set_title("TT")
    ax[0][1].set_title("EE")
    ax[0][2].set_title("BB")
    ax[1][0].set_xlabel(r"$\ell$")
    ax[1][1].set_xlabel(r"$\ell$")
    ax[1][2].set_xlabel(r"$\ell$")
    ax[0][0].set_ylabel(axis_labels[0])
    ax[1][0].set_ylabel(axis_labels[1])
    ax[0][2].legend(bbox_to_anchor=(1.1, 1.05), fancybox=True, shadow=True)
    ax[0][0].loglog()
    ax[0][1].loglog()
    ax[0][2].loglog()
    ax[1][0].set_xscale("log")
    ax[1][1].set_xscale("log")
    ax[1][2].set_xscale("log")
    ax[1][0].grid(axis="y", c="k", alpha=0.5, ls="dashed")
    ax[1][1].grid(axis="y", c="k", alpha=0.5, ls="dashed")
    ax[1][2].grid(axis="y", c="k", alpha=0.5, ls="dashed")
    if lims_x is None:
        lims_x = (2, ell[-1])
        # ell[-1] allows to avoid empty space on the right,
        # 2 is to avoid the first 2 ell that are ill-defined
    ax[0][0].set_xlim(lims_x)
    ax[0][0].set_ylim(lims_y)

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(plot_dir / save_name, bbox_inches="tight")
    plt.close()


def plotTTEEBB(
    plot_dir,
    freqs,
    Cl,
    save_name,
    legend_labels=(r"fg $C_\ell$ $\nu=$",),
    y_axis_label="y_axis",
    use_D_ell=True,
    lims_x=(2, 2000),
    lims_y=(1e-2, 1e7),
):
    """ """

    ell = np.arange(0, Cl.shape[-1])
    norm = ell * (ell + 1) / 2 / np.pi

    if not use_D_ell:
        norm = 1

    fig, ax = plt.subplots(1, 3, sharex=True, sharey="row", figsize=(16, 9))
    for f in range(Cl.shape[0]):
        ax[0].plot(ell, norm * Cl[f, 0], color="C" + str(f), ls="-", alpha=0.4)
        ax[1].plot(ell, norm * Cl[f, 1], color="C" + str(f), ls="-", alpha=0.4)
        ax[2].plot(
            ell,
            norm * Cl[f, 2],
            label=legend_labels[0] + str(freqs[f]) * (Cl.shape[0] != 1),
            color="C" + str(f),
            ls="-",
            alpha=0.4,
        )

    ax[0].set_title("TT")
    ax[1].set_title("EE")
    ax[2].set_title("BB")
    if lims_x is None:
        lims_x = (2, ell[-1])
    ax[0].set_xlim(lims_x)
    ax[0].set_ylim(lims_y)

    ax[0].set_xlabel(r"$\ell$")
    ax[1].set_xlabel(r"$\ell$")
    ax[2].set_xlabel(r"$\ell$")

    ax[0].set_ylabel(y_axis_label)

    ax[2].legend(bbox_to_anchor=(1.1, 1.05), fancybox=True, shadow=True)
    ax[0].loglog()
    ax[1].loglog()
    ax[2].loglog()
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(plot_dir / save_name, bbox_inches="tight")
    plt.close()


def plot_fg_sims(manager, config, maps=True, cls=True):
    plot_dir = manager.path_to_mock_plots
    plot_dir.mkdir(parents=True, exist_ok=True)

    fg_freq_maps = mock._generate_map_fgs_pysm(config)
    fg_freq_maps_beamed = np.zeros_like(fg_freq_maps)

    for i_f, _f in enumerate(config.frequencies):
        fg_freq_maps_beamed[i_f] = mock._beam_winpix_correction(
            config, fg_freq_maps[i_f], config.beams[i_f]
        )

    # binary_mask = hp.read_map(manager.path_to_binary_mask)
    # fg_freq_maps[..., np.where(binary_mask == 0)[0]] = hp.UNSEEN
    # fg_freq_maps_beamed[..., np.where(binary_mask == 0)[0]] = hp.UNSEEN

    fg_freq_maps = _apply_binary_mask(manager, fg_freq_maps, unseen=True)
    fg_freq_maps_beamed = _apply_binary_mask(manager, fg_freq_maps_beamed, unseen=True)
    if maps:
        cmap = cm.RdBu
        cmap.set_under("w")
        vmin = {"I": -300, "Q": -10, "U": -10}
        vmax = {"I": 300, "Q": 10, "U": 10}
        plt.figure(figsize=(20, 7))
        k = 0
        for j_stokes, stokes in enumerate(["I", "Q", "U"]):
            for i_f, fr in enumerate(config.frequencies):
                hp.mollview(
                    fg_freq_maps_beamed[i_f, j_stokes],
                    cmap=cmap,
                    title=f"{fr} GHz {stokes}",
                    min=vmin[stokes],
                    max=vmax[stokes],
                    sub=(3, len(config.frequencies), k + 1),
                )
                k += 1
        plt.savefig(plot_dir / "fg_freqs_unbeamed.png", bbox_inches="tight")
        plt.clf()
    if cls:
        cls = []
        cls_beamed = []
        for i_f, _f in enumerate(config.frequencies):
            cls.append(hp.anafast(fg_freq_maps[i_f]))
            cls_beamed.append(hp.anafast(fg_freq_maps_beamed[i_f]))
        cls = np.array(cls)
        cls_beamed = np.array(cls_beamed)
        IPython.embed()
        plotTTEEBB(
            plot_dir=plot_dir,
            freqs=config.frequencies,
            Cl=cls,
            save_name="fg_cls_unbeamed.png",
            use_D_ell=True,
            y_axis_label=r"$D_\ell$ fg unbeamed",
            lims_x=None,
            lims_y=None,
        )
        plotTTEEBB(
            plot_dir=plot_dir,
            freqs=config.frequencies,
            Cl=cls_beamed,
            save_name="fg_cls_beamed.png",
            use_D_ell=True,
            y_axis_label=r"$D_\ell$ fg beamed",
            lims_x=None,
            lims_y=None,
        )


def plot_cmb_sims(manager, config, maps=True, cls=True):
    Cl_cmb_model = mock._get_Cl_CMB_model_from_manager(manager)
    cmb_map = mock._generate_map_cmb(config, Cl_cmb_model)

    plot_dir = manager.path_to_mock_plots
    plot_dir.mkdir(parents=True, exist_ok=True)

    if maps:
        cmap = cm.RdBu
        cmap.set_under("w")
        vmin = {"I": -300, "Q": -5, "U": -5}
        vmax = {"I": 300, "Q": 5, "U": 5}
        plt.figure(figsize=(20, 7))
        for j_stokes, stokes in enumerate("IQU"):
            hp.mollview(
                cmb_map[j_stokes],
                cmap=cmap,
                title=f"CMB {stokes}",
                min=vmin[stokes],
                max=vmax[stokes],
                sub=(1, 3, j_stokes + 1),
            )
        plt.savefig(plot_dir / "cmb_maps.png", bbox_inches="tight")
        plt.clf()
    if cls:
        cls = hp.anafast(cmb_map)
        cls = np.array(cls)

        plotTTEEBB_diff(
            plot_dir=plot_dir,
            freqs=[1],
            Cl_data=cls,
            Cl_model=Cl_cmb_model,
            save_name="cmb.png",
            legend_labels=(r"CMB SIMS $DS_\ell$ $\nu=$", r"CMB INPUT $DS_\ell$ $\nu=$"),
            axis_labels=[r"$D_\ell^{\rm CMB}$", "Relative diff"],
            use_D_ell=True,
            lims_x=None,
            lims_y=None,
        )


def plot_noise_sims(manager, config, maps=True, cls=True):
    binary_mask = hp.read_map(manager.path_to_binary_mask)

    fsky_binary = sum(binary_mask) / len(binary_mask)
    nhits_map = hp.read_map(manager.path_to_nhits_map)
    nhits_map_rescaled = nhits_map / max(nhits_map)

    plot_dir = manager.path_to_mock_plots
    plot_dir.mkdir(parents=True, exist_ok=True)

    if config.noise_sim_pars.noise_option == "white_noise":
        n_ell, map_white_noise_levels = mock._get_noise(config, fsky_binary)
        noise_freq_maps = mock._get_noise_map_from_white_noise(manager, map_white_noise_levels)

    elif config.noise_sim_pars.noise_option == "noise_spectra":
        n_ell, map_white_noise_levels = mock._get_noise(config, fsky_binary)
        noise_freq_maps = mock._get_noise_map_from_noise_spectra(manager, n_ell)

    if maps:
        cmap = cm.RdBu
        cmap.set_under("w")
        vmin = {"I": -2, "Q": -0.5, "U": -0.5}
        vmax = {"I": 2, "Q": 0.5, "U": 0.5}
        plt.figure(figsize=(20, 7))
        k = 0
        for j_stokes, stokes in enumerate(["I", "Q", "U"]):
            for i_f, fr in enumerate(config.frequencies):
                hp.mollview(
                    noise_freq_maps[i_f, j_stokes],
                    cmap=cmap,
                    title=f"{fr} GHz {stokes}",
                    min=vmin[stokes],
                    max=vmax[stokes],
                    sub=(3, len(config.frequencies), k + 1),
                )
                k += 1

        plt.savefig(plot_dir / "noise_freq_maps.png", bbox_inches="tight")
        plt.clf()
    if cls:
        cls = []
        for i_f, _f in enumerate(config.frequencies):
            cls.append(hp.anafast(noise_freq_maps[i_f]))
        cls = np.array(cls)

        if config.noise_sim_pars.include_nhits:
            fsky_correction = (
                nhits_map_rescaled[..., np.where(binary_mask == 1)[0]].mean() * fsky_binary
            )
        else:
            fsky_correction = 1.0

        if config.noise_sim_pars.noise_option == "white_noise":
            cl_model = np.ones_like(cls)
            cl_model[:, 0] = (
                (map_white_noise_levels[:, np.newaxis] / np.sqrt(2) * np.pi / 180 / 60) ** 2
                * fsky_binary
                / fsky_correction
            )
            cl_model[:, 1] = (
                (map_white_noise_levels[:, np.newaxis] * np.pi / 180 / 60) ** 2
                * fsky_binary
                / fsky_correction
            )
            cl_model[:, 2] = (
                (map_white_noise_levels[:, np.newaxis] * np.pi / 180 / 60) ** 2
                * fsky_binary
                / fsky_correction
            )
        elif config.noise_sim_pars.noise_option == "noise_spectra":
            cl_model = np.zeros_like(cls)
            cl_model[:, 1, 2:-1] = n_ell * fsky_binary / fsky_correction
            cl_model[:, 2, 2:-1] = n_ell * fsky_binary / fsky_correction
        plotTTEEBB_diff(
            plot_dir=plot_dir,
            freqs=config.frequencies,
            Cl_data=cls,
            Cl_model=cl_model,
            save_name="noise_spectra.png",
            lims_x=None,
            lims_y=None,
            legend_labels=[r"noise spectra $\nu=$", r"noise model $\nu=$"],
            use_D_ell=False,
            axis_labels=[r"$C_\ell^{\rm{noise}}$", "Relative diff"],
        )


def plot_saved_sims(manager, config, maps=True, cls=True):
    plot_dir = manager.path_to_mock_plots
    plot_dir.mkdir(parents=True, exist_ok=True)

    combined_maps = _read_input_maps(manager)
    combined_maps = _apply_binary_mask(manager, combined_maps, unseen=True)

    if maps:
        cmap = cm.RdBu
        cmap.set_under("w")
        vmin = {"I": -300, "Q": -10, "U": -10}
        vmax = {"I": 300, "Q": 10, "U": 10}
        plt.figure(figsize=(20, 7))
        k = 0
        for j_stokes, stokes in enumerate(["I", "Q", "U"]):
            for i_f, fr in enumerate(config.frequencies):
                hp.mollview(
                    combined_maps[i_f, j_stokes],
                    cmap=cmap,
                    title=f"{fr} GHz {stokes}",
                    min=vmin[stokes],
                    max=vmax[stokes],
                    sub=(3, len(config.frequencies), k + 1),
                )
                k += 1

        plt.savefig(plot_dir / "combined_map.png", bbox_inches="tight")
        plt.clf()
    if cls:
        cls = []
        for i_f, _f in enumerate(config.frequencies):
            cls.append(hp.anafast(combined_maps[i_f]))
        cls = np.array(cls)

        plotTTEEBB(
            plot_dir=plot_dir,
            freqs=config.frequencies,
            Cl=cls,
            save_name="combined_cls.png",
            lims_x=(2, 200),
            lims_y=(1e-3, 1e5),
        )


def main():
    parser = argparse.ArgumentParser(description="Plotter for mask_hanlder output")
    parser.add_argument("--config", type=Path, help="config file")
    args = parser.parse_args()
    if args.config is None:
        logger.warning("No config file provided, using example config")
        config = Config.get_example()
    else:
        config = Config.from_yaml(args.config)
    manager = DataManager(config)
    manager.dump_config()

    logger.info("Plotting mocker outputs...")
    timer = Timer()
    timer.start("mock_plotter")

    plot_fiducial_spectra(manager)
    plot_fg_sims(manager, config)
    plot_cmb_sims(manager, config)
    plot_noise_sims(manager, config)
    plot_saved_sims(manager, config)

    timer.stop("mock_plotter", "Plotting mock outputs")


if __name__ == "__main__":
    main()
