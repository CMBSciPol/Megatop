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

    if map_set.shape == (1, 3, map_set.shape[-1]) or map_set.shape == (1, 2, map_set.shape[-1]):
        enum_freq = [0]
        row = 1
        column = map_set.shape[1]
    elif map_set.shape == (len(config.frequencies), 3, map_set.shape[-1]) or map_set.shape == (
        len(config.frequencies),
        2,
        map_set.shape[-1],
    ):
        enum_freq = config.frequencies
        row = map_set.shape[1]
        column = len(config.frequencies)
    else:
        logger.error(
            f"In freq_maps_plotter() map_set doesn't have the right shape, must be (nfreq, nstokes, npix) OR (1, nstokes, npix), here: {map_set.shape}"
        )
        msg = "Bad map set shape"
        raise TypeError(msg)

    stokes_list = ["I", "Q", "U"] if map_set.shape[1] == 3 else ["Q", "U"]

    for j_stokes, stokes in enumerate(stokes_list):
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


def freq_maps_plotter_one_stoke(
    config,
    map_set,
    plot_dir,
    plot_name,
    vmin=None,
    vmax=None,
    cmap=cm.RdBu,
    cmap_set_under="w",
    title_prefix="",
):
    """
    This function plots the frequency maps. It directly saves the plot directly.

    Args:
        config: The configuration object.
        map_set (ndarray): The frequency maps, with shape (num_freq, 1, num_pixels).
        plot_dir (str): The path to the directory where the plot will be saved.
        plot_name (str): The name of the file to save the plot.
        vmin (dict): The minimum values for the colorbar of the plots.
        vmax (dict): The maximum values for the colorbar of the plots.
        cmap (colormap): The colormap to use for the plots.
        cmap_set_under (str): The color to set for the values under vmin.


    Returns:
        None
    """

    cmap.set_under(cmap_set_under)

    plt.figure(figsize=(20, 7))

    # TODO: it nfreq>6 this might get a bit busy...
    row = 1
    column = len(config.frequencies)

    for i_f, fr in enumerate(config.frequencies):
        title_map = f"{title_prefix} {fr} GHz"

        hp.mollview(
            map_set[i_f],
            cmap=cmap,
            title=title_map,
            min=vmin,
            max=vmax,
            sub=(row, column, i_f + 1),
        )

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


def plot_all_Cls(all_Cls, bin_centre, plot_dir, plot_name, use_D_ell=True, y_axis_label="y_axis"):
    """
    This function plots the Cls outputed from map_to_cl or noise_spectra_estimator.
    It directly saves the plot directly.

    Args:
        all_Cls (dict): Each entry correspond to auto or cross correlation between
                        component maps (e.g "CMBxCMB", "CMBxDust" etc)
                        in the shape of (num_spectra, num_ell) where num_spectra=4 : [EE,EB,BE,BB]
        bin_centre (ndarray): The bin centres.
        plot_dir (str): The path to the directory where the plot will be saved.
        plot_name (str): The name of the file to save the plot.
        use_D_ell (bool): If True, the Cls are multiplied by ell*(ell+1)/2/pi.
        y_axis_label (str): The label for the y axis of the plot.

    Returns:
        None
    """
    norm = bin_centre * (bin_centre + 1) / 2 / np.pi
    if not use_D_ell:
        norm = 1
    # Define the labels
    labels = ["EE", "EB", "BE", "BB"]

    # Create the figure
    fig, ax = plt.subplots(1, 4, sharex=True, sharey="row", figsize=(16, 9))

    ax = ax.flatten()

    # Loop over the different power spectra
    for i, label in enumerate(labels):
        # Loop over the different components
        for j, key in enumerate(all_Cls.keys()):
            alpha = 0.4 if key not in ("CMBxCMB", "Noise_CMBxNoise_CMB") else 1
            color = "C" + str(j) if key not in ("CMBxCMB", "Noise_CMBxNoise_CMB") else "black"
            ax[i].scatter(
                bin_centre, norm * all_Cls[key][i], marker=".", label=key, color=color, alpha=alpha
            )
            ax[i].scatter(
                bin_centre,
                -norm * all_Cls[key][i],
                marker="x",
                label="-" + key,
                color=color,
                alpha=alpha,
            )

        ax[i].set_title(label)

        ax[i].set_yscale("log")
        ax[i].set_xscale("log")
        ax[i].set_xlabel(r"$\ell$")
    ax[0].set_ylabel(y_axis_label)
    ax[3].legend(bbox_to_anchor=(1.1, 1.05), fancybox=True, shadow=True)

    plt.subplots_adjust(wspace=0, hspace=0)

    plt.savefig(plot_dir / plot_name, bbox_inches="tight")
    plt.close()


def plot_all_Cls_diff(
    all_Cls,
    bin_centre,
    cls_model,
    plot_dir,
    plot_name,
    use_D_ell=True,
    y_axis_label="y_axis",
    # , beam_dict=None
):
    norm = bin_centre * (bin_centre + 1) / 2 / np.pi
    if not use_D_ell:
        norm = 1
    # Define the labels
    labels = ["EE", "EB", "BE", "BB"]

    # Create the figure
    # beam_key = "eff_all"
    fig, ax = plt.subplots(4, 4, sharex=True, sharey="row", figsize=(16, 9))

    ax = ax.flatten()

    # Loop over the different power spectra
    for i, label in enumerate(labels):
        # Loop over the different components
        for j, key in enumerate(all_Cls.keys()):
            alpha = 0.4 if key not in ("CMBxCMB", "Noise_CMBxNoise_CMB") else 1
            color = "C" + str(j) if key not in ("CMBxCMB", "Noise_CMBxNoise_CMB") else "black"
            ax[i].scatter(
                bin_centre, norm * all_Cls[key][i], marker=".", label=key, color=color, alpha=alpha
            )
            ax[i].scatter(
                bin_centre,
                -norm * all_Cls[key][i],
                marker="x",
                label="-" + key,
                color=color,
                alpha=alpha,
            )

            ax[i].plot(bin_centre, norm * cls_model[key][i], color=color, ls="--", alpha=alpha)

            ax[i + 4].scatter(
                bin_centre,
                (all_Cls[key][i] - cls_model[key][i]),
                marker=".",
                color=color,
                alpha=alpha,
            )
            ax[i + 4].scatter(
                bin_centre,
                -(all_Cls[key][i] - cls_model[key][i]),
                marker="x",
                color=color,
                alpha=alpha,
            )

            ax[i + 8].scatter(
                bin_centre,
                all_Cls[key][i] / cls_model[key][i],
                color=color,
                marker="o",
                alpha=alpha,
            )

            ax[i + 12].scatter(
                bin_centre,
                (all_Cls[key][i] - cls_model[key][i]) / cls_model[key][i],
                color=color,
                marker=".",
                alpha=alpha,
            )
            ax[i + 12].scatter(
                bin_centre,
                -(all_Cls[key][i] - cls_model[key][i]) / cls_model[key][i],
                color=color,
                marker="+",
                alpha=alpha,
            )

        ax[i].set_title(label)

        ax[i].set_yscale("log")
        ax[i].set_xscale("log")

        ax[i + 4].set_yscale("log")
        ax[i + 4].set_xscale("log")

        ax[i + 8].set_yscale("log")
        ax[i + 8].set_xscale("log")

        ax[i + 12].set_yscale("log")
        ax[i + 12].set_xscale("log")

        ax[i].set_xlabel(r"$\ell$")
    ax[0].set_ylabel(y_axis_label)
    ax[4].set_ylabel(r"$\Delta C_\ell$")
    ax[8].set_ylabel(r"$C_\ell^{\rm{data}} / C_\ell^{\rm{model}}$")
    ax[12].set_ylabel(r"$\Delta C_\ell / C_\ell^{\rm{model}}$")
    ax[3].legend(bbox_to_anchor=(1.1, 1.05), fancybox=True, shadow=True)

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(plot_dir / plot_name, bbox_inches="tight")
    plt.close()
