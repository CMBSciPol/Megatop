import argparse
import os
from pathlib import Path

import healpy as hp
import numpy as np

import megatop.utils.harmonic as hu
from megatop import Config, DataManager
from megatop.utils import Timer, logger
from megatop.utils.binning import load_nmt_binning
from megatop.utils.mask import apply_binary_mask
from megatop.utils.plot import (
    freq_maps_plotter,
    freq_maps_plotter_one_stoke,
    group_indices_by_experiment,
    plotTTEEBB,
)

HEALPY_DATA_PATH = os.getenv("HEALPY_LOCAL_DATA", None)


def plot_noisecov(manager, config, maps=True, cls=True):
    plot_dir = manager.path_to_covar_plots
    plot_dir.mkdir(parents=True, exist_ok=True)

    fname_noise_cov_maps = manager.path_to_pixel_noisecov
    noise_cov_maps = np.load(fname_noise_cov_maps)

    experiments = set(m.exp_tag for m in config.map_sets)
    binary_mask = {exp: hp.read_map(manager.path_to_binary_mask(exp)) for exp in experiments}

    for i_m, map_set in enumerate(config.map_sets):
        _ = apply_binary_mask(noise_cov_maps[i_m], binary_mask[map_set.exp_tag], unseen=True)

    if maps:
        diff_Q_U_maps = noise_cov_maps[:, 1] - noise_cov_maps[:, 2]
        for i_m, map_set in enumerate(config.map_sets):
            _ = apply_binary_mask(diff_Q_U_maps[i_m], binary_mask[map_set.exp_tag], unseen=True)

        relat_diff_Q_U_maps = diff_Q_U_maps / noise_cov_maps[:, 1]
        for i_m, map_set in enumerate(config.map_sets):
            _ = apply_binary_mask(relat_diff_Q_U_maps[i_m], binary_mask[map_set.exp_tag], unseen=True)


    if cls:
        lmax = config.plot_pars.lmax_plot
        spectra_array = np.array(
            [hu.anafast(noise_cov_maps[i], lmax=lmax) for i in range(len(config.map_sets))]
        )

    if config.parametric_sep_pars.use_harmonic_compsep:
        nmt_bins = load_nmt_binning(manager)
        bin_index_lminlmax = np.load(manager.path_to_binning, allow_pickle=True)[
            "bin_index_lminlmax"
        ]
        ell_bin_lminlmax = nmt_bins.get_effective_ells()[bin_index_lminlmax]
        binned_nl = np.load(manager.path_to_nl_noisecov)
        unbinned_nl = np.load(manager.path_to_nl_noisecov_unbinned)

    for exp, idx in group_indices_by_experiment(config.map_sets).items():
        exp_config = config.model_copy(update={"map_sets": [config.map_sets[i] for i in idx]})

        if maps:
            freq_maps_plotter(exp_config, noise_cov_maps[idx], plot_dir, f"noise_cov_maps_{exp}")

            freq_maps_plotter_one_stoke(
                exp_config,
                diff_Q_U_maps[idx],
                plot_dir,
                f"diff_Q_U_noise_cov_{exp}",
                title_prefix="Q-U noise cov",
            )

            freq_maps_plotter_one_stoke(
                exp_config,
                relat_diff_Q_U_maps[idx],
                plot_dir,
                f"relat_diff_Q_U_noise_cov_{exp}",
                title_prefix="(Q-U)/Q noise cov",
            )

        if cls:
            plotTTEEBB(
                plot_dir=plot_dir,
                freqs=exp_config.frequencies,
                Cl=spectra_array[idx],
                save_name=f"spectra_noise_cov_anafast_{exp}",
                y_axis_label=r"$C_\ell$ noise covariance (from anafast maps)",
                use_D_ell=False,
                lims_x=None,
                lims_y=None,
            )

        if config.parametric_sep_pars.use_harmonic_compsep:
            plotTTEEBB(
                plot_dir=plot_dir,
                freqs=exp_config.frequencies,
                Cl=binned_nl[idx],
                save_name=f"harmonic_pipe_spectra_noise_cov_binned_{exp}",
                y_axis_label=r"$N_\ell$ binned noise covariance (namaster)",
                use_D_ell=False,
                lims_x=None,
                lims_y=None,
                ell=ell_bin_lminlmax,
            )

            plotTTEEBB(
                plot_dir=plot_dir,
                freqs=exp_config.frequencies,
                Cl=unbinned_nl[idx],
                save_name=f"harmonic_pipe_spectra_noise_cov_unbinned_{exp}",
                y_axis_label=r"$N_\ell$ unbinned noise covariance (namaster)",
                use_D_ell=False,
                lims_x=None,
                lims_y=None,
                ell=None,
            )


def main():
    parser = argparse.ArgumentParser(description="Plotter for pixel noise cov output")
    parser.add_argument("--config", type=Path, help="config file")
    args = parser.parse_args()
    if args.config is None:
        logger.warning("No config file provided, using example config")
        config = Config.get_example()
    else:
        config = Config.load_yaml(args.config)
    manager = DataManager(config)
    manager.dump_config()

    logger.info("Plotting noise cov outputs...")

    with Timer("noisecov-plotter"):
        plot_noisecov(manager, config)


if __name__ == "__main__":
    main()