import argparse
import os

import fgbuster as fg
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from fgbuster.component_model import CMB, Dust, Synchrotron
from fgbuster.mixingmatrix import MixingMatrix

from megatop.utils.metadata_manager import BBmeta, Timer


def weighted_comp_sep(args):
    meta = BBmeta(args.globals)
    timer_compsep = Timer()
    timer_compsep.start("full_step")
    timer_compsep.start("loading_covmat")

    fname_covmat = os.path.join(meta.covmat_directory, "pixel_noise_cov_preprocessed.npy")
    if args.verbose:
        print(f"Loading noise covariance from {fname_covmat}")
    noise_cov = np.load(fname_covmat)
    timer_compsep.stop("loading_covmat", "Loading noise covariance", args.verbose)

    timer_compsep.start("loading_maps")
    fname_preproc_maps = os.path.join(meta.pre_process_directory, "freq_maps_preprocessed.npy")
    if args.verbose:
        print(f"Loading pre-processed frequency maps from {fname_preproc_maps} ")
    freq_maps_preprocessed = np.load(fname_preproc_maps)

    timer_compsep.stop("loading_maps", "Loading pre-processed frequency maps", args.verbose)

    timer_compsep.start("compsep")
    instrument = {"frequency": meta.frequencies}
    if meta.parametric_sep_pars["DEBUG_UseSynchrotron"]:
        components = [CMB(), Dust(150.0, temp=20.0), Synchrotron(150.0)]
        components_label_list = ["CMB", "Dust", "Synchrotron"]  # This is only used for plotting
    else:
        components = [CMB(), Dust(150.0, temp=20.0)]
        components_label_list = ["CMB", "Dust"]  # This is only used for plotting

    options = meta.parametric_sep_pars["options"]
    tol = meta.parametric_sep_pars["tol"]
    method = meta.parametric_sep_pars["method"]

    # FGBuster's weighted component separation used hp.UNSEEN to ignore masked pixels
    # If put to 0, I don't think they weigh on the outcome but it slows the process down and can result in warnings/errors
    binary_mask = meta.read_mask("binary").astype(bool)
    freq_maps_preprocessed_QU_masked = freq_maps_preprocessed[:, 1:]
    freq_maps_preprocessed_QU_masked[..., np.where(binary_mask == 0)[0]] = hp.UNSEEN

    noise_cov_QU_masked = noise_cov[:, 1:]
    noise_cov_QU_masked[..., np.where(binary_mask == 0)[0]] = hp.UNSEEN

    res = fg.separation_recipes.weighted_comp_sep(
        components,
        instrument,
        data=freq_maps_preprocessed_QU_masked,
        cov=noise_cov_QU_masked,  # Slice to remove the T maps, otherwise the separation will be biased
        options=options,
        tol=tol,
        method=method,
    )

    if args.verbose:
        print("success: ", res.success)
    if args.verbose:
        print("results: ", res.x)
    if args.verbose:
        print("results: ", res)
    timer_compsep.stop("compsep", "Component separation", args.verbose)

    A = MixingMatrix(*components)
    A_ev = A.evaluator(np.array(instrument["frequency"]))
    A_maxL = A_ev(res.x)
    res.A_maxL = A_maxL

    # test_invAtNA = np.linalg.inv(np.einsum('cf,fqp,fs->csqp', A_maxL.T, 1/noise_cov[:,1:], A_maxL).T).T
    # sanity_check = np.max(np.abs( ((test_invAtNA - res.invAtNA) / res.invAtNA * 100))[...,binary_mask])

    # test_invAtNA_U = np.dot(A_maxL.T, np.dot(1/noise_cov[:,2], A_maxL))
    # sanity_check = np.linalg.inv(A_maxL.T @ noise_cov @ A_maxL) - res.invAtNA
    W_maxL = np.einsum("ijsp, jf, fsp -> ifsp", res.invAtNA[:, :], A_maxL.T, 1 / noise_cov[:, 1:])
    res.W_maxL = W_maxL

    # Apply W to noise simulation:

    print("\n\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n")
    print("WARNING: THE APPLICATION OF W TO THE NOISE MAP IS A TEST")
    print("IT SHOULD BE APPLIED TO NOISE MAPS AFTER PRE-PROCESSING")
    print("FOR THE TEST THE PREPOCESSING DOESN'T CHANGE ANYTHING")
    print("DO NOT USE IT FOR ESTIMATING NOISE Cls IN GENERAL (FOR NOW)")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n")
    freq_noise_maps_array = []

    # maps_list = meta.maps_list
    # for m in maps_list:
    #     if args.verbose: print('Importing map: ', m)
    #     path_noise_map = meta.get_noise_map_filename(m)
    #     freq_noise_maps_array.append(hp.read_map(path_noise_map, field=None).tolist())
    # freq_noise_maps_array = np.array(freq_noise_maps_array)
    freq_noise_maps_array = np.load(
        os.path.join(meta.covmat_directory, "freq_noise_maps_preprocessed.npy")
    )

    freq_noise_maps_array = freq_noise_maps_array[:, 1:]
    noise_map_after_compsep = np.einsum("ifsp,fsp->isp", W_maxL, freq_noise_maps_array)
    noise_map_after_compsep[..., np.where(binary_mask == 0)[0]] = 0  # hp.UNSEEN

    res_dict = {}
    for attr in dir(res):
        if not attr.startswith("__"):
            res_dict[attr] = getattr(res, attr)
    np.savez(os.path.join(meta.components_directory, "comp_sep_results.npz"), **res_dict)

    # res.s and res.invAtNA are saved twice, but they are the direct needed outputs for the next step
    # space could be saved by adding an if statement in the above dict construction (TODO?)
    np.save(os.path.join(meta.components_directory, "components_maps.npy"), res.s)
    np.save(os.path.join(meta.components_directory, "invAtNA.npy"), res.invAtNA)
    np.save(
        os.path.join(meta.components_directory, "noise_map_after_compsep.npy"),
        noise_map_after_compsep,
    )

    if args.plots:
        timer_compsep.start("plotting")
        components_results_plotting(res, meta, components_label_list, noise_map_after_compsep)
        timer_compsep.stop("plotting", "Plotting", args.verbose)

    timer_compsep.stop("full_step", "Full component separation step", args.verbose)
    return res


def components_results_plotting(
    res, meta, components_label_list=["CMB", "Dust", "Synchrotron"], noise_map_after_compsep=None
):
    binary_mask = meta.read_mask("binary").astype(bool)
    res.s[..., np.where(binary_mask == 0)[0]] = hp.UNSEEN

    plot_dir = meta.plot_dir_from_output_dir(meta.components_directory_rel)

    fig = plt.figure(figsize=(12, 12))
    for i, component_label in enumerate(components_label_list):
        for j, stokes_label in enumerate(["Q", "U"]):
            hp.mollview(
                res.s[i, j],
                title=component_label + " " + stokes_label,
                sub=(3, 2, (2 * i + j) + 1),
                fig=fig,
                cbar=True,
            )
    plt.savefig(plot_dir + "/components_maps.png")
    plt.close()

    res.invAtNA[..., np.where(binary_mask == 0)[0]] = hp.UNSEEN

    fig = plt.figure(figsize=(12, 12))
    for i, component_label in enumerate(components_label_list):
        for j, stokes_label in enumerate(["Q", "U"]):
            hp.mollview(
                res.invAtNA[i, i, j],
                title="Noise " + component_label + "--" + stokes_label + " -- norm = log",
                sub=(3, 2, (2 * i + j) + 1),
                fig=fig,
                cbar=True,
                norm="log",
            )
    plt.savefig(plot_dir + "/noise_per_components_maps.png")
    plt.close()

    if noise_map_after_compsep is not None:
        fig = plt.figure(figsize=(12, 12))
        noise_map_after_compsep[..., np.where(binary_mask == 0)[0]] = hp.UNSEEN
        for i, component_label in enumerate(components_label_list):
            for j, stokes_label in enumerate(["Q", "U"]):
                hp.mollview(
                    noise_map_after_compsep[i, j],
                    title="Noise " + component_label + "--" + stokes_label,
                    sub=(3, 2, (2 * i + j) + 1),
                    fig=fig,
                    cbar=True,
                )
        plt.savefig(plot_dir + "/noise_maps_after_compsep.png")
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="simplistic simulator")  # TODO change name ??
    parser.add_argument("--globals", type=str, help="Path to yaml with global parameters")
    parser.add_argument("--plots", action="store_true", help="Plot the generated maps if True.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    res = weighted_comp_sep(args)
