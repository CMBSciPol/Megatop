import argparse
import os

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np

from megatop.utils import BBmeta
from megatop.utils.logger import logger


def plot_all_maps(meta, map_array, output, file_name):
    """
    Plot all maps in dictionary

    Parameters:
    -----------
    meta : BBmeta
        Metadata object
    map_array : numpy array
        Array of maps to plot, shape (n_freq, n_stokes, n_pix)
    output : str
        Path to save the plots
    file_name : str
        Name of the file to save

    Returns:
    --------
    None
    """
    # ploting Q and U maps for all entries in dictionary
    line_number = len(map_array)
    stokes = ["T", "Q", "U"]
    j = 1
    plt.figure(figsize=(6, 12))

    binary_mask = meta.read_mask("binary").astype(bool)
    for i, map_name in enumerate(meta.maps_list):
        for s in range(1, 3):
            map_plot = map_array[i][s]
            map_plot[np.where(binary_mask == 0)[0]] = hp.UNSEEN
            hp.mollview(
                map_plot, title=f"{map_name} {stokes[s]}", sub=(line_number, 2, j)
            )  # , norm='hist')
            j += 1
    plt.savefig(os.path.join(output, f"{file_name}.png"), dpi=400, bbox_inches="tight")
    plt.close()


def plot_all_spectra(spectra_dict, output, file_name):
    """
    Plot all spectra in dictionary

    Parameters:
    -----------
    spectra_dict : dict
        Dictionary containing TT,EE,BB,TE spectra to be plotted for a given frequency
    output : str
        Path to save the plots

    Returns:
    --------
    None
    """
    # ploting Q and U maps for all entries in dictionary
    line_number = len(spectra_dict.keys())
    stokes = ["TT", "EE", "BB", "TE"]
    j = 1

    fig, ax = plt.subplots(line_number, figsize=(6, 12))
    if line_number == 1:
        ax = [ax]  # to avoid problems with indexing when only one key is present
    for i, key in enumerate(spectra_dict.keys()):
        for j in range(1, 3):
            ax[i].plot(spectra_dict[key][j], label=stokes[j])
            ax[i].set_yscale("log")
            ax[i].set_xscale("log")
            ax[i].legend()
            ax[i].set_title(key)

    plt.savefig(os.path.join(output, f"{file_name}.png"), dpi=400, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="simplistic simulator")  # TODO change name ?
    parser.add_argument("--globals", type=str, help="Path to yaml with global parameters")

    args = parser.parse_args()

    meta = BBmeta(args.globals)

    meta.timer.start("plots_noise_cov")

    logger.info("\n\nPlotting Noise Covariance outputs\n\n")

    fname = os.path.join(meta.covmat_directory, "pixel_noise_cov_preprocessed.npy")
    noise_cov_maps = np.load(fname)

    plot_dir = meta.plot_dir_from_output_dir(meta.covmat_directory_rel)

    # Plotting the maps
    plot_all_maps(meta, noise_cov_maps, plot_dir, "noise_cov_maps")

    spectra_dict = {}
    lmax = 3 * meta.nside

    for i, map_name in enumerate(meta.maps_list):
        spectra_dict[map_name] = hp.anafast(noise_cov_maps[i], lmax=lmax)
    plot_all_spectra(spectra_dict, plot_dir, "check_spectra_noise_cov")
    meta.timer.stop("plots_noise_cov", "Noise covariance outputs plots")

    logger.info("\n\nPlotting Noise Covariance outputs completed succesfully\n\n")


if __name__ == "__main__":
    main()
