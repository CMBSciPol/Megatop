import argparse
from pathlib import Path

import fgbuster as fg
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from fgbuster.component_model import CMB, Dust, Synchrotron
from fgbuster.mixingmatrix import MixingMatrix

from megatop import Config
from megatop.utils import Timer, logger


def weighted_comp_sep(config: Config):
    timer = Timer()
    timer.start("full_step")

    timer.start("loading_covmat")
    noisecov_fname = config.path_to_pixel_noisecov
    logger.debug(f"Loading covmat from {noisecov_fname}")
    noisecov = np.load(noisecov_fname)
    timer.stop("loading_covmat", "Loading noise covariance")

    timer.start("loading_maps")
    preproc_maps_fname = config.get_path_to_preprocessed_maps()
    logger.debug(f"Loading input maps from {preproc_maps_fname}")
    freq_maps_preprocessed = np.load(preproc_maps_fname)
    timer.stop("loading_maps", "Loading pre-processed frequency maps")

    timer.start("compsep")
    instrument = {"frequency": config.frequencies}
    if config.parametric_sep_pars.include_synchrotron:
        components = [CMB(), Dust(150.0, temp=20.0), Synchrotron(150.0)]
        # component_labels = ["CMB", "Dust", "Synchrotron"]  # This is only used for plotting
    else:
        components = [CMB(), Dust(150.0, temp=20.0)]
        # component_labels = ["CMB", "Dust"]  # This is only used for plotting

    # get the 'options' through the appropriate method which returns a dict
    options = config.parametric_sep_pars.get_minimize_options_as_dict()
    tol = config.parametric_sep_pars.minimize_tol
    method = config.parametric_sep_pars.minimize_method

    # FGBuster's weighted component separation used hp.UNSEEN to ignore masked pixels
    # If put to 0, I don't think they weigh on the outcome but it slows the process down and can result in warnings/errors
    binary_mask = hp.read_map(config.path_to_binary_mask).astype(bool)
    freq_maps_preprocessed_QU_masked = freq_maps_preprocessed[:, 1:]
    freq_maps_preprocessed_QU_masked[..., np.where(binary_mask == 0)[0]] = hp.UNSEEN

    noisecov_QU_masked = noisecov[:, 1:]
    noisecov_QU_masked[..., np.where(binary_mask == 0)[0]] = hp.UNSEEN

    res = fg.separation_recipes.weighted_comp_sep(
        components,
        instrument,
        data=freq_maps_preprocessed_QU_masked,
        cov=noisecov_QU_masked,
        options=options,
        tol=tol,
        method=method,
    )

    # if args.verbose:
    #     print("success: ", res.success)
    # if args.verbose:
    #     print("results: ", res.x)
    # if args.verbose:
    #     print("results: ", res)
    # timer.stop("compsep", "Component separation")

    A = MixingMatrix(*components)
    A_ev = A.evaluator(np.array(instrument["frequency"]))
    A_maxL = A_ev(res.x)  # pyright: ignore[reportCallIssue]
    res.A_maxL = A_maxL

    logger.info(f"Success: {res.success} -> {res.message}")
    logger.info(f"Spectral parameters {res.params} -> {res.x}")
    timer.stop("compsep", "Component separation (FGbuster weighted compsep)")

    # test_invAtNA = np.linalg.inv(np.einsum('cf,fqp,fs->csqp', A_maxL.T, 1/noise_cov[:,1:], A_maxL).T).T
    # sanity_check = np.max(np.abs( ((test_invAtNA - res.invAtNA) / res.invAtNA * 100))[...,binary_mask])

    # test_invAtNA_U = np.dot(A_maxL.T, np.dot(1/noise_cov[:,2], A_maxL))
    # sanity_check = np.linalg.inv(A_maxL.T @ noise_cov @ A_maxL) - res.invAtNA
    # W_maxL = np.einsum("ijsp, jf, fsp -> ifsp", res.invAtNA[:, :], A_maxL.T, 1 / noise_cov[:, 1:])
    # res.W_maxL = W_maxL

    # # Apply W to noise simulation:

    # print("\n\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n")
    # print("WARNING: THE APPLICATION OF W TO THE NOISE MAP IS A TEST")
    # print("IT SHOULD BE APPLIED TO NOISE MAPS AFTER PRE-PROCESSING")
    # print("FOR THE TEST THE PREPOCESSING DOESN'T CHANGE ANYTHING")
    # print("DO NOT USE IT FOR ESTIMATING NOISE Cls IN GENERAL (FOR NOW)")
    # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n")
    # freq_noise_maps_array = []

    # # maps_list = meta.maps_list
    # # for m in maps_list:
    # #     if args.verbose: print('Importing map: ', m)
    # #     path_noise_map = meta.get_noise_map_filename(m)
    # #     freq_noise_maps_array.append(hp.read_map(path_noise_map, field=None).tolist())
    # # freq_noise_maps_array = np.array(freq_noise_maps_array)
    # freq_noise_maps_array = np.load(
    #     os.path.join(meta.covmat_directory, "freq_noise_maps_preprocessed.npy")
    # )

    # freq_noise_maps_array = freq_noise_maps_array[:, 1:]
    # noise_map_after_compsep = np.einsum("ifsp,fsp->isp", W_maxL, freq_noise_maps_array)
    # noise_map_after_compsep[..., np.where(binary_mask == 0)[0]] = 0  # hp.UNSEEN

    # res_dict = {}
    # for attr in dir(res):
    #     if not attr.startswith("__"):
    #         res_dict[attr] = getattr(res, attr)
    # np.savez(os.path.join(meta.components_directory, "comp_sep_results.npz"), **res_dict)

    # res.s and res.invAtNA are saved twice, but they are the direct needed outputs for the next step
    # space could be saved by adding an if statement in the above dict construction (TODO?)
    config.path_to_components.mkdir(parents=True, exist_ok=True)
    config.path_to_covar.mkdir(parents=True, exist_ok=True)
    np.save(config.path_to_components_maps, res.s)
    np.save(config.path_to_invAtNA, res.invAtNA)

    # np.save(
    #     os.path.join(meta.components_directory, "noise_map_after_compsep.npy"),
    #     noise_map_after_compsep,
    # )

    # if args.plots:
    #     timer.start("plotting")
    #     components_results_plotting(res, meta, components_label_list, noise_map_after_compsep)
    #     timer.stop("plotting", "Plotting")

    timer.stop("full_step", "Full component separation step")
    return res


def components_results_plotting(
    res,
    config: Config,
    component_labels=("CMB", "Dust", "Synchrotron"),
    noise_map_after_compsep=None,
):
    binary_mask = hp.read_map(config.path_to_binary_mask).astype(bool)
    res.s[..., np.where(binary_mask == 0)[0]] = hp.UNSEEN

    plot_dir = config.path_to_components_plots
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(12, 12))
    for i, component_label in enumerate(component_labels):
        for j, stokes_label in enumerate("QU"):
            hp.mollview(
                res.s[i, j],
                title=component_label + " " + stokes_label,
                sub=(3, 2, (2 * i + j) + 1),
                fig=fig,
                cbar=True,
            )
    # plt.savefig(plot_dir / "components_maps.png")
    # plt.close()

    res.invAtNA[..., np.where(binary_mask == 0)[0]] = hp.UNSEEN

    fig = plt.figure(figsize=(12, 12))
    for i, component_label in enumerate(component_labels):
        for j, stokes_label in enumerate("QU"):
            hp.mollview(
                res.invAtNA[i, i, j],
                title="Noise " + component_label + "--" + stokes_label + " -- norm = log",
                sub=(3, 2, (2 * i + j) + 1),
                fig=fig,
                cbar=True,
                norm="log",
            )
    # plt.savefig(plot_dir / "noise_per_components_maps.png")
    # plt.close()
    plt.show()

    if noise_map_after_compsep is not None:
        fig = plt.figure(figsize=(12, 12))
        noise_map_after_compsep[..., np.where(binary_mask == 0)[0]] = hp.UNSEEN
        for i, component_label in enumerate(component_labels):
            for j, stokes_label in enumerate("QU"):
                hp.mollview(
                    noise_map_after_compsep[i, j],
                    title="Noise " + component_label + "--" + stokes_label,
                    sub=(3, 2, (2 * i + j) + 1),
                    fig=fig,
                    cbar=True,
                )
        plt.savefig(plot_dir / "noise_maps_after_compsep.png")
        plt.close()


def save_compsep_results(config: Config, res):
    fname_results = config.path_to_compsep_results
    res_dict = {}
    for attr in dir(res):
        if not attr.startswith("__"):
            res_dict[attr] = getattr(res, attr)
    # Saving result dict
    logger.info(f"Saving compsep results to {fname_results}")
    np.savez(fname_results, **res_dict)
    # Saving component maps
    fname_compmaps = config.path_to_components_maps
    logger.info(f"Saving component maps to {fname_compmaps}")
    np.save(fname_compmaps, res.s)


def main():
    parser = argparse.ArgumentParser(description="Perform the component separation")
    parser.add_argument("--config", type=Path, help="config file")
    args = parser.parse_args()
    config = Config.from_yaml(args.config)
    config.dump()
    res = weighted_comp_sep(config)
    save_compsep_results(config, res)
    # components_results_plotting(res, meta)


if __name__ == "__main__":
    main()
