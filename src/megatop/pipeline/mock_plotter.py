import argparse
import os

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

from megatop.utils import BBmeta, mock_utils


def plot_fiducial_spectra(meta):
    plot_dir = meta.plot_dir_from_output_dir("sims")

    path_Cl_BB_lens = meta.get_fname_cls_fiducial_cmb("lensed")
    path_Cl_BB_prim_r1 = meta.get_fname_cls_fiducial_cmb("unlensed_scalar_tensor_r1")

    Cl_lens = hp.read_cl(path_Cl_BB_lens)
    Cl_prim = hp.read_cl(path_Cl_BB_prim_r1)[..., : Cl_lens.shape[-1]]
    Cl_BB_prim = meta.map_sim_pars["r_input"] * Cl_prim[2]
    Cl_BB_lens = meta.map_sim_pars["A_lens_input"] * Cl_lens[2]
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
    plt.savefig(os.path.join(plot_dir, "fiducial_CMB_spectra.png"), bbox_inches="tight")
    plt.clf()


def plotTTEEBB_diff(
    plot_dir,
    freqs,
    Cl_data,
    Cl_model,
    save_name,
    legend_labels=[r"label data $C_\ell$ $\nu=$", r"label model $C_\ell$ $\nu=$"],
    axis_labels=["y_axis_row0", "y_axis_row1"],
    use_D_ell=True,
    lims={"x": (2, 2000), "y": (1e-2, 1e7)},
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
            label=legend_labels[0] + str(freqs[f]) * (Cl_data.shape[0] != 1),  #
            color="C" + str(f),
            ls="-",
            alpha=0.4,
        )
        ax[0][0].plot(ell, norm * Cl_model[f, 0], color="C" + str(f), ls=":")
        ax[0][1].plot(ell, norm * Cl_model[f, 1], color="C" + str(f), ls=":")
        ax[0][2].plot(
            ell,
            norm * Cl_model[f, 2],
            label=legend_labels[1] + str(freqs[f]) * (Cl_data.shape[0] != 1),  #
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
    ax[1][0].set_xlabel(r"\ell")
    ax[1][1].set_xlabel(r"\ell")
    ax[1][2].set_xlabel(r"\ell")
    ax[0][0].set_ylabel(axis_labels[0])
    ax[1][0].set_ylabel(axis_labels[1])
    ax[0][2].legend(bbox_to_anchor=(1.1, 1.05), fancybox=True, shadow=True)
    ax[0][0].loglog()
    ax[0][1].loglog()
    ax[0][2].loglog()
    ax[1][0].set_xscale("log")
    ax[1][1].set_xscale("log")
    ax[1][2].set_xscale("log")
    ax[0][0].set_xlim(lims["x"])
    ax[0][0].set_ylim(lims["y"])
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(os.path.join(plot_dir, save_name), bbox_inches="tight")
    plt.close()


def plotTTEEBB(
    plot_dir,
    freqs,
    Cl,
    save_name,
    legend_labels=[r"fg $C_\ell$ $\nu=$"],
    use_D_ell=True,
    lims={"x": (2, 2000), "y": (1e-2, 1e7)},
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
            label=legend_labels[0] + str(freqs[f]) * (Cl.shape[0] != 1),  #
            color="C" + str(f),
            ls="-",
            alpha=0.4,
        )

    ax[0].set_title("TT")
    ax[1].set_title("EE")
    ax[2].set_title("BB")
    ax[0].set_xlim(lims["x"])
    ax[0].set_ylim(lims["y"])

    ax[2].legend(bbox_to_anchor=(1.1, 1.05), fancybox=True, shadow=True)
    ax[0].loglog()
    ax[1].loglog()
    ax[2].loglog()
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(os.path.join(plot_dir, save_name), bbox_inches="tight")
    plt.close()


def plot_fg_sims(meta, maps=True, cls=True):
    fg_freqs_maps = mock_utils.generate_map_fgs_pysm(meta)
    fg_freqs_maps_beamed = np.zeros_like(fg_freqs_maps)
    beams = [91.0, 63.0, 30.0, 17.0, 11.0, 9.0]
    for i_f, f in enumerate(meta.frequencies):
        fg_freqs_maps_beamed[i_f] = mock_utils.beam_winpix_correction(
            meta, fg_freqs_maps[i_f], beams[i_f]
        )
    binary_mask = meta.read_mask("binary")
    fg_freqs_maps[..., np.where(binary_mask == 0)[0]] = hp.UNSEEN
    fg_freqs_maps_beamed[..., np.where(binary_mask == 0)[0]] = hp.UNSEEN
    if maps:
        cmap = cm.RdBu
        cmap.set_under("w")
        vmin = {"I": -300, "Q": -10, "U": -10}
        vmax = {"I": 300, "Q": 10, "U": 10}
        plt.figure(figsize=(20, 7))
        k = 0
        for j_stokes, stokes in enumerate(["I", "Q", "U"]):
            for i_f, fr in enumerate(meta.frequencies):
                hp.mollview(
                    fg_freqs_maps_beamed[i_f, j_stokes],
                    cmap=cmap,
                    title=f"{fr} GHz {stokes}",
                    min=vmin[stokes],
                    max=vmax[stokes],
                    sub=(3, len(meta.frequencies), k + 1),
                )
                k += 1
        plot_dir = meta.plot_dir_from_output_dir("sims")
        plt.savefig(os.path.join(plot_dir, "fg_freqs_unbeamed.png"), bbox_inches="tight")
        plt.clf()
    if cls:
        cls = []
        cls_beamed = []
        for i_f, f in enumerate(meta.frequencies):
            cls.append(hp.anafast(fg_freqs_maps[i_f]))
            cls_beamed.append(hp.anafast(fg_freqs_maps_beamed[i_f]))
        cls = np.array(cls)
        cls_beamed = np.array(cls_beamed)
        plot_dir = meta.plot_dir_from_output_dir("sims")
        plotTTEEBB(
            plot_dir=plot_dir,
            freqs=meta.frequencies,
            Cl=cls,
            save_name="fg_cls_unbeamed.png",
            lims={"x": (2, 500), "y": (1e-2, 1e7)},
        )
        plotTTEEBB(
            plot_dir=plot_dir,
            freqs=meta.frequencies,
            Cl=cls_beamed,
            save_name="fg_cls_beamed.png",
            lims={"x": (2, 500), "y": (1e-2, 1e7)},
        )


def plot_cmb_sims(meta, maps=True, cls=True):
    Cl_cmb_model = mock_utils.get_Cl_CMB_model_from_meta(meta)
    cmb_map = mock_utils.generate_map_cmb(meta, Cl_cmb_model)
    if maps:
        cmap = cm.RdBu
        cmap.set_under("w")
        vmin = {"I": -300, "Q": -10, "U": -10}
        vmax = {"I": 300, "Q": 10, "U": 10}
        plt.figure(figsize=(20, 7))
        k = 0
        for j_stokes, stokes in enumerate(["I", "Q", "U"]):
            hp.mollview(
                cmb_map[j_stokes],
                cmap=cmap,
                title=f"CMB {stokes}",
                min=vmin[stokes],
                max=vmax[stokes],
                sub=(1, 3, k + 1),
            )
            k += 1
        plot_dir = meta.plot_dir_from_output_dir("sims")
        plt.savefig(os.path.join(plot_dir, "cmb_maps.png"), bbox_inches="tight")
        plt.clf()
    if cls:
        cls = hp.anafast(cmb_map)
        cls = np.array(cls)
        plot_dir = meta.plot_dir_from_output_dir("sims")
        plotTTEEBB_diff(
            plot_dir=plot_dir,
            freqs=[1],
            Cl_data=cls,
            Cl_model=Cl_cmb_model,
            save_name="cmb.png",
            lims={"x": (2, 500), "y": (1e-2, 1e7)},
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="simplistic simulator")
    parser.add_argument("--globals", type=str, help="Path to yaml with global parameters")
    args = parser.parse_args()
    meta = BBmeta(args.globals)
    plot_fiducial_spectra(meta)
    plot_fg_sims(meta)
    plot_cmb_sims(meta)
