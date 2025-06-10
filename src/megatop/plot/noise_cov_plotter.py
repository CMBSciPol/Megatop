import argparse
from pathlib import Path

import healpy as hp
import numpy as np

from megatop import Config, DataManager
from megatop.utils import Timer, logger
from megatop.utils.mask import apply_binary_mask
from megatop.utils.plot import freq_maps_plotter, freq_maps_plotter_one_stoke, plotTTEEBB


def plot_noisecov(manager, config, maps=True, cls=True):
    plot_dir = manager.path_to_covar_plots
    plot_dir.mkdir(parents=True, exist_ok=True)

    fname_noise_cov_maps = manager.path_to_pixel_noisecov
    noise_cov_maps = np.load(fname_noise_cov_maps)
    binary_mask = hp.read_map(manager.path_to_binary_mask)

    noise_cov_maps = apply_binary_mask(noise_cov_maps, binary_mask, unseen=True)

    if maps:
        freq_maps_plotter(config, noise_cov_maps, plot_dir, "noise_cov_maps")

        diff_Q_U_maps = noise_cov_maps[:, 1] - noise_cov_maps[:, 2]
        diff_Q_U_maps = apply_binary_mask(diff_Q_U_maps, binary_mask, unseen=True)

        freq_maps_plotter_one_stoke(
            config,
            diff_Q_U_maps,
            plot_dir,
            "diff_Q_U_noise_cov",
            title_prefix="Q-U noise cov",
        )

        relat_diff_Q_U_maps = diff_Q_U_maps / noise_cov_maps[:, 1]
        relat_diff_Q_U_maps = apply_binary_mask(relat_diff_Q_U_maps, binary_mask, unseen=True)

        freq_maps_plotter_one_stoke(
            config,
            relat_diff_Q_U_maps,
            plot_dir,
            "relat_diff_Q_U_noise_cov",
            title_prefix="(Q-U)/Q noise cov",
        )

    if cls:
        lmax = 3 * config.nside
        spectra_array = []
        for i in range(len(config.frequencies)):
            spectra_array.append(hp.anafast(noise_cov_maps[i], lmax=lmax))
        spectra_array = np.array(spectra_array)

        plotTTEEBB(
            plot_dir=plot_dir,
            freqs=config.frequencies,
            Cl=spectra_array,
            save_name="spectra_noise_cov_anafast",
            y_axis_label=r"$C_\ell$ noise covariance (from anafast maps)",
            use_D_ell=False,
            lims_x=None,
            lims_y=None,
        )

    if config.parametric_sep_pars.use_harmonic_compsep:
        ell_bin_lminlmax = np.load(manager.path_to_effectiv_bins_harmonic_compsep)
        binned_nl = np.load(manager.path_to_nl_noisecov)
        unbinned_nl = np.load(manager.path_to_nl_noisecov_unbinned)
        plotTTEEBB(
            plot_dir=plot_dir,
            freqs=config.frequencies,
            Cl=binned_nl,
            save_name="harmonic_pipe_spectra_noise_cov_binned",
            y_axis_label=r"$N_\ell$ binned noise covariance (namaster)",
            use_D_ell=False,
            lims_x=None,
            lims_y=None,
            ell=ell_bin_lminlmax,
        )

        plotTTEEBB(
            plot_dir=plot_dir,
            freqs=config.frequencies,
            Cl=unbinned_nl,
            save_name="harmonic_pipe_spectra_noise_cov_unbinned",
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
