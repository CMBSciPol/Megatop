import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

from megatop.utils import logger


def freq_maps_plotter(
    config,
    map_set,
    plot_dir,
    plot_name,
    vmin=None,
    vmax=None,
    cmap=cm.RdBu,
    cmap_set_under="w",
    component="CMB",
):
    """
    This function plots the frequency maps. It directly saves the plot directly.

    Args:
        config: The configuration object.
        map_set (ndarray): The frequency maps, with shape (num_freq, num_stokes, num_pixels) or (1, num_stokes, num_pixels) if
                            there is only one set of (TQU) map to display (e.g. CMB maps).
        plot_dir (str): The path to the directory where the plot will be saved.
        plot_name (str): The name of the file to save the plot.
        vmin (dict): The minimum values for the colorbar of the plots.
        vmax (dict): The maximum values for the colorbar of the plots.
        cmap (colormap): The colormap to use for the plots.
        cmap_set_under (str): The color to set for the values under vmin.
        component (str): If map_set of the shape (1, num_stokes, num_pixels) this parameter gives the
                        name of the component to plot, e.g. CMB, fg, etc.

    Returns:
        None
    """

    if vmin is None:
        vmin = {"I": None, "Q": None, "U": None}
    if vmax is None:
        vmax = {"I": None, "Q": None, "U": None}

    cmap.set_under(cmap_set_under)

    plt.figure(figsize=(20, 7))
    k = 0

    if map_set.shape == (1, 3, map_set.shape[-1]):
        enum_freq = [0]
        row = 1
        column = 3
    elif map_set.shape == (len(config.frequencies), 3, map_set.shape[-1]):
        enum_freq = config.frequencies
        row = 3
        column = len(config.frequencies)
    else:
        logger.error(
            f"In freq_maps_plotter() map_set doesn't have the right shape, must be (nfreq, nstokes, npix) OR (1, nstokes, npix), here: {map_set.shape}"
        )
        msg = "Bad map set shape"
        raise TypeError(msg)

    for j_stokes, stokes in enumerate(["I", "Q", "U"]):
        for i_f, fr in enumerate(enum_freq):
            title_map = f"{component} {stokes}" if enum_freq == [0] else f"{fr} GHz {stokes}"

            hp.mollview(
                map_set[i_f, j_stokes],
                cmap=cmap,
                title=title_map,
                min=vmin[stokes],
                max=vmax[stokes],
                sub=(row, column, k + 1),
            )
            k += 1

    plt.savefig(plot_dir / plot_name, bbox_inches="tight")
    plt.clf()


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
    """
    This function plots the Cls. It directly saves the plot directly.

    Args:
        plot_dir (str): The path to the directory where the plot will be saved.
        freqs (list): The frequencies of the maps.
        Cl (ndarray): The Cls, with shape (num_freq, num_spectra [TT,EE,BB], num_ell).
        save_name (str): The name of the file to save the plot.
        legend_labels (list): The labels for the legend of the plot.
        y_axis_label (str): The label for the y axis of the plot.
        use_D_ell (bool): If True, the Cls are multiplied by ell*(ell+1)/2/pi.
        lims_x (tuple): The limits for the x axis of the plot.
        lims_y (tuple): The limits for the y axis of the plot.
    """

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

    ax[0].set_xlabel(r"$\ell$")
    ax[1].set_xlabel(r"$\ell$")
    ax[2].set_xlabel(r"$\ell$")

    ax[0].set_ylabel(y_axis_label)

    ax[2].legend(bbox_to_anchor=(1.1, 1.05), fancybox=True, shadow=True)
    ax[0].loglog()
    ax[1].loglog()
    ax[2].loglog()

    ax[0].set_xlim(lims_x)
    ax[0].set_ylim(lims_y)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(plot_dir / save_name, bbox_inches="tight")
    plt.close()


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
        plot_dir (str): The path to the directory where the plot will be saved.
        freqs (list): The frequencies of the maps.
        Cl_data (ndarray): The Cls of the data, with shape (num_freq, num_spectra [TT,EE,BB], num_ell).
        Cl_model (ndarray): The Cls of the model that the data will be compared to,
                            with shape (num_freq, num_spectra [TT,EE,BB], num_ell).
        save_name (str): The name of the file to save the plot.
        legend_labels (list): The labels for the legend of the plot (one for data, one for model)
        axis_labels (list): The labels for the y axis of the plot.
        use_D_ell (bool): If True, the Cls are multiplied by ell*(ell+1)/2/pi.
        lims_x (tuple): The limits for the x axis of the plot.
        lims_y (tuple): The limits for the y axis of the plot.


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
