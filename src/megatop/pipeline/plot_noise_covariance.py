import argparse
import os

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np

from megatop.utils.metadata_manager import BBmeta, Timer


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="simplistic simulator")  # TODO change name ?
    parser.add_argument("--globals", type=str, help="Path to yaml with global parameters")
    parser.add_argument("--plots", action="store_true", help="Plot the generated maps if True.")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    if not args.plots:
        exit()

    timer_plots = Timer()
    timer_plots.start("plots_noise_cov")

    print("\n\nPlotting Noise Covariance outputs\n\n")

    meta = BBmeta(args.globals)

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
    timer_plots.stop("plots_noise_cov", "Noise covariance outputs plots", args.verbose)

    print("\n\nPlotting Noise Covariance outputs completed succesfully\n\n")


'''

# =================================================================================
# =                     Plotting functions and wrappers                           =
# =================================================================================


def plot_cov_matrix(
    args, noise_cov_mean, file_name, mask_unseen=None, norm=None, minmax=(None, None)
):
    """
    Plots the noise covariance matrix for each frequency map. Saves it directly to the output plot directory.

    Parameters
    ----------
    args : argparse.Namespace
        The arguments from the command line.
    noise_cov_mean : np.ndarray
        The noise covariance matrix.
    file_name : str
        The name of the file to save the plot.
    mask_unseen : np.ndarray, optional
        The mask to apply to the plot. The default is None.
    norm : str, optional
        The normalization of the plot. The default is None.

    Returns
    -------
    None.
    """

    meta = BBmeta(args.globals)

    plot_dir = meta.plot_dir_from_output_dir(meta.covmat_directory_rel)
    cmap = cm.RdBu
    cmap.set_under("w")

    cols = math.ceil(
        math.sqrt(noise_cov_mean.shape[0])
    )  # making a grid to display all the frequency maps
    rows = math.ceil(noise_cov_mean.shape[0] / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(16, 9))

    for ax, f in zip(axes.flatten(), range(noise_cov_mean.shape[0])):
        plt.sca(ax)  # setting current axe
        noise_cov_mean_ = noise_cov_mean[f].copy()
        if mask_unseen is not None:
            noise_cov_mean_[np.where(mask_unseen == 0)[0]] = hp.UNSEEN

        try:
            hp.mollview(
                noise_cov_mean_,
                cmap=cmap,
                cbar=True,
                hold=True,
                title=rf"Noise cov map $\nu={meta.frequencies[f]}$ GHz",
                norm=norm,
                min=minmax[0],
                max=minmax[1],
            )
        except ValueError as e:
            print("WARNING catching ERROR: ", e)
            print("Noise cov map could not be plotted for frequency ", meta.frequencies[f])
            print(
                "This is likely due to choice of norm=", norm, " with negative values in the map."
            )
            print("Please check the input data.")
            print("ERROR HANDLED by setting norm=None (default)")
            hp.mollview(
                noise_cov_mean_,
                cmap=cmap,
                cbar=True,
                hold=True,
                title=rf"Noise cov map $\nu={meta.frequencies[f]}$ GHz",
                norm=None,
                min=minmax[0],
                max=minmax[1],
            )
            continue
        hp.graticule()

    map_noise_cov_save_path = os.path.join(plot_dir, file_name)
    plt.savefig(map_noise_cov_save_path)
    plt.close()


def plot_hist_freqmaps(
    args, freq_maps, save_name, plot_gauss=False, max_y_from_gauss=False, bins=100, binary_mask=None
):
    """
    Plots histograms of frequency maps. Saves it directly to the output plot directory.

    Args:
        args (object): The arguments object.
        freq_maps (ndarray): The frequency maps.
        save_name (str): The name of the file to save the plot.
        plot_gauss (bool, optional): Whether to plot Gaussian curves. Defaults to False.
        bins (int, optional): The number of bins in the histograms. Defaults to 100.
        binary_mask (ndarray, optional): The binary mask. Defaults to None.

    Returns:
        None
    """

    meta = BBmeta(args.globals)

    if binary_mask is None:
        binary_mask = np.ones(freq_maps.shape[-1], dtype=bool)

    fig, ax = plt.subplots(1, 3, figsize=(16, 9))
    for f in range(len(meta.frequencies)):
        ax[0].hist(freq_maps[f, 0, binary_mask], bins=bins, histtype="step", density=True)
        ax[1].hist(freq_maps[f, 1, binary_mask], bins=bins, histtype="step", density=True)
        ax[2].hist(
            freq_maps[f, 2, binary_mask],
            bins=bins,
            histtype="step",
            label=r"$\nu = $" + str(meta.frequencies[f]) + " GHz",
            density=True,
        )
    if plot_gauss:
        x = np.linspace(-5, 5, 1000)

        ax[0].plot(x, 1 / np.sqrt(2 * np.pi) * np.exp(-(x**2) / 2))
        ax[1].plot(x, 1 / np.sqrt(2 * np.pi) * np.exp(-(x**2) / 2))
        ax[2].plot(x, 1 / np.sqrt(2 * np.pi) * np.exp(-(x**2) / 2), label=r"$\mathcal{N}(0,1) $")

        if max_y_from_gauss:
            max_y = 5 * np.max(1 / np.sqrt(2 * np.pi) * np.exp(-(x**2) / 2))

            ax[0].set_ylim(0, max_y)
            ax[1].set_ylim(0, max_y)
            ax[2].set_ylim(0, max_y)

    plt.legend()

    ax[0].set_title("T")
    ax[1].set_title("Q")
    ax[2].set_title("U")

    plot_dir = meta.plot_dir_from_output_dir(meta.covmat_directory_rel)
    plt.savefig(os.path.join(plot_dir, save_name))
    plt.close()


def wrapper_plotting(
    args,
    meta,
    rank,
    noise_cov_mean,
    noise_cov_preprocessed_mean,
    freq_noise_maps_array,
    freq_noise_maps_pre_processed_array,
    binary_mask_from_nhits_nside_sims,
):
    """
    Wrapper for plotting the noise covariance matrix and the frequency maps before and after pre-processing in the case where
    simulations are loaded from disk (i.e.  meta.noise_cov_pars['noise_cov_method'] == 'load_sims').
    This will:
        - check the white noise level of the mean noise covariance
        - create a noise simulation from the mean noise covariance
        - compute the std and plot the spectra of the noise simulation / sqrt(noise cov) (should be close to unity)
        - plot the histograms of the noise simulation / sqrt(noise cov) spectra
        - plots the spectra of the noise covariance against the theoretical power spectra from V3calc
        - plots the spectra from the noise simulation generated from the noise covariance against the theoretical power spectra from V3calc
        - the same ratio is done with one of the input noise simulation used for the computation of the noise covariance matrix
        - spectra and histograms of the input noise simulation / sqrt(noise cov) are plotted before and after pre-processing
          in the latter case some correction to the spectra is applied to account for the pre-processing. However this is NOT well understood yet.
          DON'T TRUST THIS PLOT. (plot name: 'ratio_noisecov_preprocessed_map_SIM2'+str(i)+'.png',)
    Saves the plots directly to the output plot directory.


    Parameters
    ----------
    args : argparse.Namespace
        The arguments from the command line.
    meta : object
        The metadata manager object from BBmeta.
    rank : int
        The rank of the process.
    noise_cov_mean : np.ndarray
        The noise covariance matrix.
    noise_cov_preprocessed_mean : np.ndarray
        The noise covariance matrix after pre-processing.
    freq_noise_maps_array : np.ndarray
        The frequency noise maps.
    freq_noise_maps_pre_processed_array : np.ndarray
        The frequency noise maps after pre-processing.
    binary_mask_from_nhits_nside_sims : np.ndarray
        The binary mask from the nhits map.

    Returns
    -------
    binary_mask_from_nhits_preproc : np.ndarray
        The binary mask from the nhits map after pre-processing.

    """

    print("\nPLOTTING...\n")
    meta_sims = BBmeta(args.sims)
    nsims = meta_sims.general_pars["nsims"]
    fsky_from_binary_mask_from_nhits_nside_sims = sum(binary_mask_from_nhits_nside_sims) / len(
        binary_mask_from_nhits_nside_sims
    )

    MemoryUsage(args, f"rank = {rank} ")

    freq_noise_maps_pre_processed_array = np.array(freq_noise_maps_pre_processed_array)
    freq_noise_maps_array = np.array(freq_noise_maps_array)

    mask = meta.read_mask("binary").astype(bool)
    fsky_mask = sum(mask) / len(mask)

    # ===========================================================================================
    # = CHECKING AND PLOTTING NOISE COVARIANCE MATRIX AGAINST THEORY AND NOISE SIM MADE FROM IT =
    # ===========================================================================================

    # Checking average white noise level on noise covariance BEFORE preprocessing
    print("=======================================")
    print("CHECK WHITE NOISE LVL FULL NHITS PATCH")
    std_uK_arcmin_noise_cov_mean_FULLPATCH = CheckWhiteNoiselvl(
        args, noise_cov_mean, fsky_mask, mask=binary_mask_from_nhits_nside_sims
    )
    np.save(
        os.path.join(
            meta.plot_dir_from_output_dir(meta.covmat_directory_rel),
            "std_uK_arcmin_noise_cov_mean_FULLPATCH.png",
        ),
        std_uK_arcmin_noise_cov_mean_FULLPATCH,
    )

    # Creating noise sim from noise_cov_mean directly for sanity check
    noise_sim_from_cov = np.random.normal(
        0 * noise_cov_mean, np.sqrt(noise_cov_mean), noise_cov_mean.shape
    )
    ratio_noisecov_sim_from_cov, cl_ratio_noisecov_sim_from_cov = CheckNoiseSpectra(
        args, noise_cov_mean, noise_sim_from_cov, mask=binary_mask_from_nhits_nside_sims
    )

    std_ratio_noisecov_sim_from_cov_FULLPATCH = np.std(
        ratio_noisecov_sim_from_cov[..., binary_mask_from_nhits_nside_sims]
        if binary_mask_from_nhits_nside_sims is not None
        else ratio_noisecov_sim_from_cov,
        axis=-1,
    )
    np.save(
        os.path.join(
            meta.plot_dir_from_output_dir(meta.covmat_directory_rel),
            "std_ratio_noisecov_sim_from_cov_FULLPATCH.png",
        ),
        std_ratio_noisecov_sim_from_cov_FULLPATCH,
    )
    print(
        "std of ratio, should be close to 1 FULL NHITS PATCH = \n",
        std_ratio_noisecov_sim_from_cov_FULLPATCH,
    )

    # Plotting spectra of noise sim from cov / sqrt(noise cov) spectra BEFORE pre-processing, including corrective factor
    plot_allspectra(
        args,
        cl_ratio_noisecov_sim_from_cov
        / hp.nside2resol(meta_sims.nside) ** 2
        / fsky_from_binary_mask_from_nhits_nside_sims,
        "ratio_noisecov_map_SIM_FROM_COV.png",
        legend_labels=[r"data $C_\ell$ $\nu=$", r"model $C_\ell$ $\nu=$"],
        axis_labels=[r"$C_\ell \, [\mu K \, rad]^2$", r"$C_\ell \, [\mu K \, rad]^2$"],
    )

    # plotting histograms of noise sim from cov / sqrt(noise cov) spectra BEFORE pre-processing

    plot_hist_freqmaps(
        args,
        ratio_noisecov_sim_from_cov,  # [...,binary_mask_from_nhits_nside_sims] if binary_mask_from_nhits_nside_sims is not None else ratio_noisecov_sim_from_cov,
        "hist_ratio_noisecov_map_SIM_FROM_COV.png",
        plot_gauss=True,
        binary_mask=binary_mask_from_nhits_nside_sims,
    )

    # Computing frequency spectra of noise_cov_mean, noise_sim_from_cov
    cl_noise_cov_FREQ_mean = []
    cl_noise_sim_from_cov_FREQ = []
    for f in range(len(meta.frequencies)):
        cl_noise_cov_FREQ_mean.append(
            hp.anafast(noise_cov_mean[f], lmax=3 * meta.general_pars["nside"], pol=True)
        )
        cl_noise_sim_from_cov_FREQ.append(
            hp.anafast(noise_sim_from_cov[f], lmax=3 * meta.general_pars["nside"], pol=True)
        )
    cl_noise_cov_FREQ_mean = np.array(cl_noise_cov_FREQ_mean)
    cl_noise_sim_from_cov_FREQ = np.array(cl_noise_sim_from_cov_FREQ)

    print(
        "plotTTEEBB_diff and get_Nl_white_noise are no longer available in the pipeline due to lack of check in preprocessing (for now?)"
    )
    # TODO: once checks are re-implemented in pre-processing, this function should be updated?

    # Noise_ell = get_Nl_white_noise(args, fsky_binary=fsky_mask)
    # min_ell_max = min(Noise_ell.shape[-1],cl_noise_sim_from_cov_FREQ.shape[-1])
    # Plotting the difference between spectra of the noise simulation generated from noise_cov_mean and the theoretical power spectra from V3calc
    # plotTTEEBB_diff(args,
    #                 cl_noise_sim_from_cov_FREQ[...,:min_ell_max],
    #                 Noise_ell[...,:min_ell_max],
    #                 os.path.join(meta.plot_dir_from_output_dir(meta.covmat_directory_rel),
    #                             'noise_SIMFromCov_vs_model_cl.png'),
    #                 legend_labels=[r'$C_\ell$ $\nu=$', r'Noise model $C_\ell$ $\nu=$'],
    #                 axis_labels=[r'$C_\ell \, [\mu K \, rad]^2$', r'$\frac{\Delta_{\ell}}{\rm{Input}_\ell}$'],
    #                 use_D_ell=False)

    # Plotting the difference between spectra of the noise_cov_mean and the theoretical power spectra from V3calc
    # plotTTEEBB_diff(args,
    #                 cl_noise_cov_FREQ_mean[...,:min_ell_max],# * (fsky_mask*4*np.pi)**2,
    #                 Noise_ell[...,:min_ell_max],
    #                 os.path.join(meta.plot_dir_from_output_dir(meta.covmat_directory_rel),
    #                             'noise_cov_mean_vs_model_cl.png'),
    #                 legend_labels=[r'$D_\ell$ $\nu=$', r'cov $D_\ell$ $\nu=$'],
    #                 axis_labels=[r'$D_\ell \, [\mu K \, rad]^2$', r'$\frac{\Delta_{\ell}}{\rm{Input}_\ell}$'],
    #                 use_D_ell=False)

    # ==================================================================================================
    # =     CHECKING AND PLOTTING NOISE COVARIANCE MATRIX AGAINST INPUT SIMS BEFORE PREPROCESSING      =
    # ==================================================================================================
    MemoryUsage(args, f"rank = {rank} ")

    # Plotting noise sim / sqrt(noise cov) spectra BEFORE pre-processing
    nsims_plot = meta.noise_cov_pars["nsims_plot"]  # number of sims to plot
    if nsims_plot > nsims:
        nsims_plot = nsims
    for i in range(nsims_plot):
        ratio_noisecov_sim, cl_ratio_noisecov_sim = CheckNoiseSpectra(
            args, noise_cov_mean, freq_noise_maps_array[i], mask=binary_mask_from_nhits_nside_sims
        )

        # Checking the std of the ratio of noise sim / sqrt(noise cov) spectra BEFORE pre-processing
        std_ratio_NoiseSim_vs_noise_cov_mean_FULLPATCH = np.std(
            ratio_noisecov_sim[..., binary_mask_from_nhits_nside_sims]
            if binary_mask_from_nhits_nside_sims is not None
            else ratio_noisecov_sim,
            axis=-1,
        )
        np.save(
            os.path.join(
                meta.plot_dir_from_output_dir(meta.covmat_directory_rel),
                "std_ratio_NoiseSim_vs_noise_cov_mean_FULLPATCH_SIM#" + str(i) + ".png",
            ),
            std_ratio_NoiseSim_vs_noise_cov_mean_FULLPATCH,
        )
        print(
            "Std of ratio before pre-processing SIM#" + str(i) + ", should be close to 1 = \n",
            std_ratio_NoiseSim_vs_noise_cov_mean_FULLPATCH,
        )

        # Plotting histograms of noise sim / sqrt(noise cov) spectra BEFORE pre-processing
        plot_hist_freqmaps(
            args,
            ratio_noisecov_sim,
            "hist_ratio_noisecov_map_SIM" + str(i) + ".png",
            plot_gauss=True,
            binary_mask=binary_mask_from_nhits_nside_sims,
        )

        # Plotting noise sim / sqrt(noise cov) spectra BEFORE pre-processing (full patch)
        plot_allspectra(
            args,
            cl_ratio_noisecov_sim
            / hp.nside2resol(meta_sims.nside) ** 2
            / fsky_from_binary_mask_from_nhits_nside_sims,
            "ratio_noisecov_map_SIM" + str(i) + ".png",
            legend_labels=[r"label data $C_\ell$ $\nu=$", r"label model $C_\ell$ $\nu=$"],
            axis_labels=[r"$C_\ell \, [\mu K \, rad]^2$", r"$C_\ell \, [\mu K \, rad]^2$"],
        )

    # ==================================================================================================
    # =      CHECKING AND PLOTTING NOISE COVARIANCE MATRIX AGAINST INPUT SIMS AFTER PREPROCESSING      =
    # ==================================================================================================

    # Plotting noise sim / sqrt(noise cov) spectra AFTER pre-processing

    # Computing effective beam and pixel window function for each frequency
    # TODO: Import from a saved effective beam generated as output from pre_processing
    lmax_convolution = 3 * meta.general_pars["nside"]
    wpix_in = hp.pixwin(
        meta_sims.nside, pol=True, lmax=lmax_convolution
    )  # Pixel window function of input maps
    wpix_out = hp.pixwin(
        meta.nside, pol=True, lmax=lmax_convolution
    )  # Pixel window function of output maps
    wpix_in[1][0:2] = 1.0  # in order not to divide by 0
    Bl_gauss_common = hp.gauss_beam(
        np.radians(meta.pre_proc_pars["common_beam_correction"] / 60),
        lmax=lmax_convolution,
        pol=True,
    )

    effective_beam_freq = []
    for f in range(len(meta.frequencies)):
        Bl_gauss_fwhm = hp.gauss_beam(
            np.radians(meta.pre_proc_pars["fwhm"][f] / 60), lmax=lmax_convolution, pol=True
        )

        bl_correction = Bl_gauss_common / Bl_gauss_fwhm

        sm_corr_T = bl_correction[:, 0] * wpix_out[0] / wpix_in[0]
        sm_corr_P = bl_correction[:, 1] * wpix_out[1] / wpix_in[1]
        effective_beam_freq.append(
            [
                sm_corr_T,
                sm_corr_P,
                sm_corr_P,
                np.sqrt(sm_corr_T * sm_corr_P),
                np.sqrt(sm_corr_P * sm_corr_P),
                np.sqrt(sm_corr_T * sm_corr_P),
            ]
        )
    effective_beam_freq = np.array(effective_beam_freq)

    # ============ This is for testing but is seems useless, to be removed =================
    # Importing mask from nhits map and applying it to noise_cov_preprocessed_mean,

    binary_mask_from_nhits_preproc = meta.read_mask("binary").astype(bool)
    noise_cov_preprocessed_mean *= binary_mask_from_nhits_preproc

    fsky_from_binary_mask_from_nhits_preproc = sum(binary_mask_from_nhits_preproc) / len(
        binary_mask_from_nhits_preproc
    )
    # TODO: Check if second mask application really useful

    print("=======================================")
    print("CHECK WHITE NOISE AFTER PRE-PROCESSING LVL FULL NHITS PATCH")
    std_uK_arcmin_noise_cov_mean_AFTERPREPOC_FULLPATCH = CheckWhiteNoiselvl(
        args, noise_cov_preprocessed_mean, fsky_mask, mask=binary_mask_from_nhits_preproc
    )
    np.save(
        os.path.join(
            meta.plot_dir_from_output_dir(meta.covmat_directory_rel),
            "std_uK_arcmin_noise_cov_mean_AFTERPREPOC_FULLPATCH.png",
        ),
        std_uK_arcmin_noise_cov_mean_AFTERPREPOC_FULLPATCH,
    )

    MemoryUsage(args, f"rank = {rank} ")

    #  ========================================================================================
    for i in range(nsims_plot):
        ratio_noisecov_sim_preprocessed, cl_ratio_noisecov_preprocessed_sim = CheckNoiseSpectra(
            args,
            noise_cov_preprocessed_mean,
            freq_noise_maps_pre_processed_array[i],
            mask=binary_mask_from_nhits_preproc,
        )

        # Checking the std of the ratio of noise sim / sqrt(noise cov) spectra AFTER pre-processing on full patch
        std_ratio_NoiseSim_vs_noise_cov_mean_PREPROCESSED_FULLPATCH = np.std(
            ratio_noisecov_sim_preprocessed[..., binary_mask_from_nhits_preproc]
            if binary_mask_from_nhits_preproc is not None
            else ratio_noisecov_sim_preprocessed,
            axis=-1,
        )
        print(
            "Std of ratio after pre-processing SIM#" + str(i) + ", should be close to 1 = \n",
            std_ratio_NoiseSim_vs_noise_cov_mean_PREPROCESSED_FULLPATCH,
        )
        np.save(
            os.path.join(
                meta.plot_dir_from_output_dir(meta.covmat_directory_rel),
                "std_ratio_NoiseSim_vs_noise_cov_mean_PREPROCESSED_FULLPATCH_SIM#"
                + str(i)
                + ".png",
            ),
            std_ratio_NoiseSim_vs_noise_cov_mean_PREPROCESSED_FULLPATCH,
        )

        # Plotting histograms of noise sim / sqrt(noise cov) spectra AFTER pre-processing on full patch and centre patch
        plot_hist_freqmaps(
            args,
            ratio_noisecov_sim_preprocessed,
            "hist_ratio_noisecov_preprocessed_map_SIM" + str(i) + ".png",
            plot_gauss=True,
            binary_mask=binary_mask_from_nhits_preproc,
        )

        # Plotting noise sim / sqrt(noise cov) spectra AFTER pre-processing (full patch) with corrective factors taking into account the effective beam and pixel window function and fsky

        ratio_noisecov_preprocessed_map_SIM = (
            cl_ratio_noisecov_preprocessed_sim
            / hp.nside2resol(meta.nside) ** 2
            / fsky_from_binary_mask_from_nhits_preproc
        )
        ratio_noisecov_preprocessed_map_SIM[..., 2:] /= effective_beam_freq[..., 2:] ** 2
        ratio_noisecov_preprocessed_map_SIM[..., :2] = 0
        plot_allspectra(
            args,
            ratio_noisecov_preprocessed_map_SIM,  # * mean_beam_offset_ell**4,
            "ratio_noisecov_preprocessed_map_SIM2" + str(i) + ".png",
            legend_labels=[r"label data $C_\ell$ $\nu=$", r"label model $C_\ell$ $\nu=$"],
            axis_labels=[r"$C_\ell \, [\mu K \, rad]^2$", r"$C_\ell \, [\mu K \, rad]^2$"],
        )
    return binary_mask_from_nhits_preproc


def wrapper_plotting_matrices_and_maps(
    args,
    meta,
    rank,
    noise_cov_mean,
    noise_cov_preprocessed_mean,
    binary_mask_from_nhits_nside_sims,
    binary_mask_from_nhits_preproc,
):
    """
    Wrapper for plotting the noise covariance matrix and the frequency maps before and after pre-processing in the case where
    simulations are loaded from disk (i.e.  meta.noise_cov_pars['noise_cov_method'] == 'load_sims').
    This will:
        - plot the noise covariance matrix
        - plot the noise covariance matrix after pre-processing
        - plot the ratio of the noise covariance matrix after pre-processing over the noise covariance matrix before pre-processing
        - plot the binary masks from the nhits maps before and after pre-processing

    Saves the plots directly to the output plot directory.

    Parameters
    ----------
    args : argparse.Namespace
        The arguments from the command line.
    meta : object
        The metadata manager object from BBmeta.
    rank : int
        The rank of the process.
    noise_cov_mean : np.ndarray
        The noise covariance matrix.
    noise_cov_preprocessed_mean : np.ndarray
        The noise covariance matrix after pre-processing.
    binary_mask_from_nhits_nside_sims : np.ndarray
        The binary mask from the nhits map before pre-processing.
    binary_mask_from_nhits_preproc : np.ndarray
        The binary mask from the nhits map after pre-processing.

    Returns
    -------
    None.
    """

    MemoryUsage(args, f"rank = {rank} ")

    # ==================================================================================================
    # =                             PLOTTING RESULTING NOISE COVARIANCE MAPS                           =
    # ==================================================================================================

    add_test_param_in_save_name = ""

    if not meta.noise_cov_pars["include_nhits"]:
        add_test_param_in_save_name += "_nonhits"
        norm_maps = None
    else:
        norm_maps = None  # 'hist'
        # TODO add norm_maps as a parameter in the noise_cov_pars? Or just remove this.

    plot_cov_matrix(
        args,
        noise_cov_mean[:, 0],
        "map_noise_cov_T" + add_test_param_in_save_name + ".png",
        norm=norm_maps,
        mask_unseen=binary_mask_from_nhits_nside_sims,
    )
    plot_cov_matrix(
        args,
        noise_cov_mean[:, 1],
        "map_noise_cov_Q" + add_test_param_in_save_name + ".png",
        norm=norm_maps,
        mask_unseen=binary_mask_from_nhits_nside_sims,
    )
    plot_cov_matrix(
        args,
        noise_cov_mean[:, 2],
        "map_noise_cov_U" + add_test_param_in_save_name + ".png",
        norm=norm_maps,
        mask_unseen=binary_mask_from_nhits_nside_sims,
    )

    plot_cov_matrix(
        args,
        noise_cov_preprocessed_mean[:, 0],
        "map_noise_cov_preprocessed_T" + add_test_param_in_save_name + ".png",
        norm=norm_maps,
        mask_unseen=binary_mask_from_nhits_preproc,
    )
    plot_cov_matrix(
        args,
        noise_cov_preprocessed_mean[:, 1],
        "map_noise_cov_preprocessed_Q" + add_test_param_in_save_name + ".png",
        norm=norm_maps,
        mask_unseen=binary_mask_from_nhits_preproc,
    )
    plot_cov_matrix(
        args,
        noise_cov_preprocessed_mean[:, 2],
        "map_noise_cov_preprocessed_U" + add_test_param_in_save_name + ".png",
        norm=norm_maps,
        mask_unseen=binary_mask_from_nhits_preproc,
    )

    noise_cov_mean_downgraded = np.array(
        [hp.ud_grade(m, nside_out=meta.nside) for m in noise_cov_mean]
    )

    ratio_noisecov_before_after_preproc = noise_cov_preprocessed_mean.copy()
    ratio_noisecov_before_after_preproc[..., binary_mask_from_nhits_preproc] /= (
        noise_cov_mean_downgraded[..., binary_mask_from_nhits_preproc]
    )
    ratio_noisecov_before_after_preproc[..., np.where(~binary_mask_from_nhits_preproc)] = 0

    for s in range(3):
        stokes_letter = ["T", "Q", "U"]
        plot_cov_matrix(
            args,
            ratio_noisecov_before_after_preproc[:, s],
            "ratio_map_noise_cov_AfterOverBeforePreprocessed_"
            + stokes_letter[s]
            + add_test_param_in_save_name
            + "_histNorm.png",
            norm=norm_maps,
            mask_unseen=binary_mask_from_nhits_preproc,
        )
        plot_cov_matrix(
            args,
            ratio_noisecov_before_after_preproc[:, s],
            "ratio_map_noise_cov_AfterOverBeforePreprocessed_"
            + stokes_letter[s]
            + add_test_param_in_save_name
            + "_linearNorm.png",
            norm=None,
            mask_unseen=binary_mask_from_nhits_preproc,
        )

    cmap = cm.RdBu
    cmap.set_under("w")
    plot_dir = meta.plot_dir_from_output_dir(meta.covmat_directory_rel)

    f = 2
    hp.mollview(
        binary_mask_from_nhits_nside_sims,
        cmap=cmap,
        cbar=True,
        hold=True,
        title=rf"Noise cov map $\nu={meta.frequencies[f]}$ GHz",
        norm=None,
    )
    hp.graticule()
    plt.savefig(os.path.join(plot_dir, "binary_mask_from_nhits_nside_sims.png"))
    plt.close()

    hp.mollview(
        binary_mask_from_nhits_preproc,
        cmap=cmap,
        cbar=True,
        hold=True,
        title=rf"Noise cov map $\nu={meta.frequencies[f]}$ GHz",
        norm=None,
    )
    hp.graticule()
    plt.savefig(os.path.join(plot_dir, "binary_mask_from_nhits_preproc.png"))
    plt.close()


def wrapper_plotting_noise_cov_from_noise_maps(
    args, meta, noise_cov_preprocessed_mean, freq_noise_maps_pre_processed, nside_in_list
):
    """
    Wrapper for plotting the noise covariance matrix and the frequency maps before and after pre-processing in the case where
    we DON'T use simulations (sims is False) but some ""true"" noise maps (e.g. for MSS2) are loaded from disk.
    This will:
        - plot the noise covariance matrix before/after pre-processing
        - computes de ratio of the preprocesed input noise map used for the computation of the noise covariance matrix over the sqrt(noise cov) matrix
        - computes the std of this ratio and save it.
        - plot the histograms of this ratio
        - plot the histograms of the noise covariance matrix with different binning
        - plot the noise sim / sqrt(noise cov) spectra (after pre-processing) with corrective factors taking into account the effective beam and pixel window function and fsky
          NOTE: This corrective factor is NOT well understood yet. DON'T TRUST THIS PLOT.
    Saves the plots directly to the output plot directory.

    Parameters
    ----------
    args : argparse.Namespace
        The arguments from the command line.
    meta : object
        The metadata manager object from BBmeta.
    noise_cov_preprocessed_mean : np.ndarray
        The noise covariance matrix after pre-processing.
    freq_noise_maps_pre_processed : np.ndarray
        The frequency noise maps after pre-processing.
    nside_in_list : list
        The list of nside for each frequency map.

    Returns
    -------
    None.

    """

    add_test_param_in_save_name = ""
    norm_maps = "log"
    binary_mask_from_nhits_preproc = meta.read_mask("binary").astype(bool)
    fsky_from_binary_mask_from_nhits_preproc = sum(binary_mask_from_nhits_preproc) / len(
        binary_mask_from_nhits_preproc
    )

    plot_cov_matrix(
        args,
        noise_cov_preprocessed_mean[:, 0],
        "map_noise_cov_preprocessed_T" + add_test_param_in_save_name + ".png",
        norm=norm_maps,
        mask_unseen=binary_mask_from_nhits_preproc,
    )
    plot_cov_matrix(
        args,
        noise_cov_preprocessed_mean[:, 1],
        "map_noise_cov_preprocessed_Q" + add_test_param_in_save_name + ".png",
        norm=norm_maps,
        mask_unseen=binary_mask_from_nhits_preproc,
    )
    plot_cov_matrix(
        args,
        noise_cov_preprocessed_mean[:, 1],
        "map_noise_cov_preprocessed_Q" + add_test_param_in_save_name + "_unmasked.png",
        norm=norm_maps,
        mask_unseen=None,
    )
    plot_cov_matrix(
        args,
        noise_cov_preprocessed_mean[:, 2],
        "map_noise_cov_preprocessed_U" + add_test_param_in_save_name + ".png",
        norm=norm_maps,
        mask_unseen=binary_mask_from_nhits_preproc,
    )

    ratio_noisecov_sim_preprocessed, cl_ratio_noisecov_preprocessed_sim = CheckNoiseSpectra(
        args,
        noise_cov_preprocessed_mean,
        freq_noise_maps_pre_processed,
        mask=binary_mask_from_nhits_preproc,
    )

    # Checking the std of the ratio of noise sim / sqrt(noise cov) spectra AFTER pre-processing on full patch
    std_ratio_NoiseSim_vs_noise_cov_mean_PREPROCESSED_FULLPATCH = np.std(
        ratio_noisecov_sim_preprocessed[..., binary_mask_from_nhits_preproc]
        if binary_mask_from_nhits_preproc is not None
        else ratio_noisecov_sim_preprocessed,
        axis=-1,
    )
    print(
        "Std of ratio after pre-processing, should be close to 1 = \n",
        std_ratio_NoiseSim_vs_noise_cov_mean_PREPROCESSED_FULLPATCH,
    )
    np.save(
        os.path.join(
            meta.plot_dir_from_output_dir(meta.covmat_directory_rel),
            "std_ratio_NoiseSim_vs_noise_cov_mean_PREPROCESSED_FULLPATCH.png",
        ),
        std_ratio_NoiseSim_vs_noise_cov_mean_PREPROCESSED_FULLPATCH,
    )
    # Plotting histograms of noise sim / sqrt(noise cov) spectra AFTER pre-processing on full patch and centre patch
    plot_hist_freqmaps(
        args,
        ratio_noisecov_sim_preprocessed,
        "hist_ratio_noisecov_preprocessed_map.png",
        plot_gauss=True,
        binary_mask=binary_mask_from_nhits_preproc,
    )

    bins = np.linspace(0, 5, 100)
    plot_hist_freqmaps(
        args,
        noise_cov_preprocessed_mean,
        "hist_noisecov_preprocessed_map_max5.png",
        plot_gauss=False,
        max_y_from_gauss=True,
        binary_mask=binary_mask_from_nhits_preproc,
        bins=bins,
    )
    plot_hist_freqmaps(
        args,
        noise_cov_preprocessed_mean,
        "hist_noisecov_preprocessed_map.png",
        plot_gauss=False,
        binary_mask=binary_mask_from_nhits_preproc,
        bins=100,
    )

    # Plotting noise sim / sqrt(noise cov) spectra AFTER pre-processing (full patch) with corrective factors taking into account the effective beam and pixel window function and fsky
    lmax_convolution = 3 * meta.general_pars["nside"]
    wpix_out = hp.pixwin(
        meta.nside, pol=True, lmax=lmax_convolution
    )  # Pixel window function of output maps
    Bl_gauss_common = hp.gauss_beam(
        np.radians(meta.pre_proc_pars["common_beam_correction"] / 60),
        lmax=lmax_convolution,
        pol=True,
    )

    effective_beam_freq = []
    for f in range(len(meta.frequencies)):
        wpix_in = hp.pixwin(
            nside_in_list[f], pol=True, lmax=lmax_convolution
        )  # Pixel window function of input maps
        wpix_in[1][0:2] = 1.0  # in order not to divide by 0

        Bl_gauss_fwhm = hp.gauss_beam(
            np.radians(meta.pre_proc_pars["fwhm"][f] / 60), lmax=lmax_convolution, pol=True
        )

        bl_correction = Bl_gauss_common / Bl_gauss_fwhm

        sm_corr_T = bl_correction[:, 0] * wpix_out[0] / wpix_in[0]
        sm_corr_P = bl_correction[:, 1] * wpix_out[1] / wpix_in[1]
        effective_beam_freq.append(
            [
                sm_corr_T,
                sm_corr_P,
                sm_corr_P,
                np.sqrt(sm_corr_T * sm_corr_P),
                np.sqrt(sm_corr_P * sm_corr_P),
                np.sqrt(sm_corr_T * sm_corr_P),
            ]
        )
    effective_beam_freq = np.array(effective_beam_freq)

    ratio_noisecov_preprocessed_map_SIM = (
        cl_ratio_noisecov_preprocessed_sim
        / hp.nside2resol(meta.nside) ** 2
        / fsky_from_binary_mask_from_nhits_preproc
    )
    ratio_noisecov_preprocessed_map_SIM[..., 2:] /= effective_beam_freq[..., 2:] ** 2
    ratio_noisecov_preprocessed_map_SIM[..., :2] = 0
    plot_allspectra(
        args,
        ratio_noisecov_preprocessed_map_SIM,  # * mean_beam_offset_ell**4,
        "ratio_noisecov_preprocessed_map.png",
        legend_labels=[r"label data $C_\ell$ $\nu=$", r"label model $C_\ell$ $\nu=$"],
        axis_labels=[r"$C_\ell \, [\mu K \, rad]^2$", r"$C_\ell \, [\mu K \, rad]^2$"],
    )


def CheckWhiteNoiselvl(args, noise_cov, fsky, mask=None):
    """
    Calculate the white noise level of noise_cov and compare it to the theoretical white noise level.

    Args:
        args: The parser arguments, containing the path to the global parameters file to set up metadata manager.
        noise_cov (ndarray): The noise covariance matrix, of shape (num_freq, num_stokes [T,Q,U], num_pix).
        fsky (float): The fraction of the sky not covered by the mask.
        mask (ndarray): The mask to apply on the noise covariance matrix, of shape (num_pix).

    Returns:
        std_uK_arcmin: The standard deviation of the noise map inside the mask, in uK arcmin.

    """

    meta = BBmeta(args.globals)

    if meta.noise_sim_pars is not None:
        ell, N_ell_P_SA, Map_white_noise_levels = V3.so_V3_SA_noise(
            sensitivity_mode=meta.noise_sim_pars["sensitivity_mode"],
            one_over_f_mode=2,  # fixed to None since we only use white noise here
            SAC_yrs_LF=meta.noise_sim_pars["SAC_yrs_LF"],
            f_sky=fsky,
            ell_max=meta.general_pars["lmax"],
            delta_ell=1,
            beam_corrected=False,
            remove_kluge=False,
            CMBS4="",
        )
    elif (
        "noise_lvl_uKarcmin" in meta.noise_cov_pars
        and meta.noise_cov_pars["noise_lvl_uKarcmin"] is not None
    ):
        Map_white_noise_levels = np.array(list(meta.noise_cov_pars["noise_lvl_uKarcmin"].values()))
    else:
        exit("ERROR: No noise level provided in noise_cov_pars or noise_sim_pars")

    # putting masked pixels to nan to use nanmean
    if mask is not None:
        if len(mask) != noise_cov.shape[-1]:
            exit("ERROR: Mask and noise_cov must have the same length")
        noise_cov_nan = noise_cov.copy()
        noise_cov_nan[..., np.where(mask == 0)[0]] = np.nan
    else:
        noise_cov_nan = noise_cov

    var_noise = np.nanmean(noise_cov_nan, axis=-1)
    std_uK_arcmin = np.sqrt(var_noise) * hp.nside2resol(
        hp.npix2nside(noise_cov_nan.shape[-1]), arcmin=True
    )

    print("std_uK_arcmin = \n", std_uK_arcmin)
    print("Map_white_noise_levels = ", Map_white_noise_levels)
    print("std_uK_arcmin / Map_white_noise_levels = ", std_uK_arcmin[:, 1] / Map_white_noise_levels)
    print(
        "(std_uK_arcmin - Map_white_noise_levels) / Map_white_noise_levels * 100 = ",
        (std_uK_arcmin[:, 1] - Map_white_noise_levels) / Map_white_noise_levels * 100,
        "\n",
    )
    return std_uK_arcmin


def CheckNoiseSpectra(args, noise_cov, noise_map, mask=None):
    """
    Calculate the noise spectra and ratio of noise map to sqrt(noise covariance).

    Args:
        args: The parser arguments, containing the path to the global parameters file to set up metadata manager.
        noise_cov (ndarray): The noise covariance matrix, of shape (num_freq, num_stokes [T,Q,U], num_pix).
        noise_map (ndarray): The noise map, of shape (num_freq, num_stokes [T,Q,U], num_pix).
        mask (ndarray): The mask to apply on the ratio_noisecov_sim of shape (num_pix).

    Returns:
        ratio_noisecov_sim: The ratio of noise map to noise covariance, of shape (num_freq, num_stokes [T,Q,U], num_pix).
        cl_ratio_noisecov_sim: The spectra of ratio_noisecov_sim, of shape (num_freq, num_spectra [TT, EE, BB, TE, EB, TB], num_ell).

    """

    meta = BBmeta(args.globals)

    ratio_noisecov_sim = np.empty(noise_map.shape)
    cl_ratio_noisecov_sim = []

    for f in range(len(meta.maps_list)):
        for s in range(noise_cov.shape[1]):
            ratio = noise_map[f, s].copy()
            ratio[np.where(noise_cov[f, s] != 0)] /= np.sqrt(
                noise_cov[f, s][np.where(noise_cov[f, s] != 0)]
            )
            ratio[np.where(noise_cov[f, s] == 0)] = 0

            ratio_noisecov_sim[f, s] = ratio
        if mask is not None:
            # Maks must be binary (boolean actually)!! simply doing *= doesn't work because some value will be np.inf and np.inf*0 = np.nan
            # which leads to problems down the line
            ratio_noisecov_sim[..., np.invert(mask)] = 0

        cl_ratio_noisecov_sim.append(
            hp.anafast(ratio_noisecov_sim[f], lmax=3 * meta.general_pars["nside"])
        )
    cl_ratio_noisecov_sim = np.array(cl_ratio_noisecov_sim)

    return ratio_noisecov_sim, cl_ratio_noisecov_sim


def plot_allspectra(
    args,
    Cl_data,
    save_name,
    legend_labels=[r"label data $C_\ell$ $\nu=$", r"label model $C_\ell$ $\nu=$"],
    axis_labels=["y_axis_row0", "y_axis_row1"],
    plot_hline_one=True,
    title=None,
    use_D_ell=False,
):
    """
    This function plots the difference between the data and the model Cls. It saves the plot directly in the plots directory of the covariance step.

    Args:
        args: The parser arguments, containing the path to the global parameters file to set up metadata manager.
        Cl_data (ndarray): The data Cls, with shape (num_freq, num_spectra [TT,EE,BB], num_ell).
        Cl_model (ndarray): The model Cls, with shape (num_freq, num_spectra [TT,EE,BB], num_ell).
        save_name (str): The name of the file to save the plot. It will save the plot in the plots directory of the simulation output directory.
                            OR complete save path if you want to save it elsewhere.
        legend_labels (list): The labels for the legend of the plot.
        axis_labels (list): The labels for the x and y axes of the plot.
        plot_hline_one (bool): If True, a horizontal line at y=1 will be plotted. Default is True.
        title (str): The main title of the plot. Default is None.

    Returns:
        None
    """

    meta = BBmeta(args.globals)

    ell = np.arange(0, Cl_data.shape[-1])
    if use_D_ell:
        norm = ell * (ell + 1) / 2 / np.pi
    else:
        norm = 1

    fig, ax = plt.subplots(2, 3, sharex=True, sharey="row", figsize=(15, 15))
    for f in range(Cl_data.shape[0]):
        ax[0][0].plot(ell, norm * Cl_data[f, 0], color="C" + str(f), ls="-", alpha=1)
        ax[0][1].plot(ell, norm * Cl_data[f, 1], color="C" + str(f), ls="-", alpha=1)
        ax[0][2].plot(
            ell,
            norm * Cl_data[f, 2],
            label=legend_labels[0] + str(meta.frequencies[f]) * (Cl_data.shape[0] != 1),  #
            color="C" + str(f),
            ls="-",
            alpha=1,
        )
        ax[1][0].plot(ell, norm * Cl_data[f, 3], color="C" + str(f), ls="-")
        ax[1][1].plot(ell, norm * Cl_data[f, 4], color="C" + str(f), ls="-")
        ax[1][2].plot(ell, norm * Cl_data[f, 5], color="C" + str(f), ls="-")

    if plot_hline_one:
        ax[0][0].hlines(1, ell[0], ell[-1], linestyles="dashed", color="black")
        ax[0][1].hlines(1, ell[0], ell[-1], linestyles="dashed", color="black")
        ax[0][2].hlines(1, ell[0], ell[-1], linestyles="dashed", color="black")
        ax[1][0].hlines(1, ell[0], ell[-1], linestyles="dashed", color="black")
        ax[1][1].hlines(1, ell[0], ell[-1], linestyles="dashed", color="black")
        ax[1][2].hlines(1, ell[0], ell[-1], linestyles="dashed", color="black")

    ax[0][0].set_title("TT")
    ax[0][1].set_title("EE")
    ax[0][2].set_title("BB")
    ax[1][0].set_title("TE")
    ax[1][1].set_title("EB")
    ax[1][2].set_title("TB")
    ax[1][0].set_xlabel(r"$\ell$")
    ax[1][1].set_xlabel(r"$\ell$")
    ax[1][2].set_xlabel(r"$\ell$")

    ax[0][0].set_ylabel(axis_labels[0])
    ax[1][0].set_ylabel(axis_labels[1])
    ax[0][2].legend(bbox_to_anchor=(1.1, 1.05), fancybox=True, shadow=True)

    ax[1][0].set_xscale("log")
    ax[1][1].set_xscale("log")
    ax[1][2].set_xscale("log")
    plt.subplots_adjust(wspace=0, hspace=0)
    fig.suptitle(title)

    plt.savefig(
        os.path.join(meta.plot_dir_from_output_dir(meta.covmat_directory_rel), save_name),
        bbox_inches="tight",
    )
    plt.close()



'''
