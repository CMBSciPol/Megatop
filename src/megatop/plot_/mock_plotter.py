import argparse
from pathlib import Path

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np

from megatop import DataManager
from megatop.config import Config
from megatop.utils import Timer, logger, mock
from megatop.utils.plot import freq_maps_plotter, plotTTEEBB, plotTTEEBB_diff
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


def plot_fg_sims(manager, config, maps=True, cls=True):
    plot_dir = manager.path_to_mock_plots
    plot_dir.mkdir(parents=True, exist_ok=True)

    fg_freq_maps = mock._generate_map_fgs_pysm(config)
    fg_freq_maps_beamed = np.zeros_like(fg_freq_maps)

    for i_f, _f in enumerate(config.frequencies):
        fg_freq_maps_beamed[i_f] = mock._beam_winpix_correction(
            config, fg_freq_maps[i_f], config.beams[i_f]
        )

    fg_freq_maps = _apply_binary_mask(manager, fg_freq_maps, unseen=True)
    fg_freq_maps_beamed = _apply_binary_mask(manager, fg_freq_maps_beamed, unseen=True)

    if maps:
        freq_maps_plotter(
            config,
            fg_freq_maps_beamed,
            plot_dir,
            "fg_freqs_unbeamed.png",
            vmin={"I": -300, "Q": -10, "U": -10},
            vmax={"I": 300, "Q": 10, "U": 10},
        )

    if cls:
        cls = []
        cls_beamed = []
        for i_f, _f in enumerate(config.frequencies):
            cls.append(hp.anafast(fg_freq_maps[i_f]))
            cls_beamed.append(hp.anafast(fg_freq_maps_beamed[i_f]))
        cls = np.array(cls)
        cls_beamed = np.array(cls_beamed)

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
        freq_maps_plotter(
            config,
            np.array([cmb_map]),
            plot_dir,
            "cmb_maps.png",
            vmin={"I": -300, "Q": -5, "U": -5},
            vmax={"I": 300, "Q": 5, "U": 5},
            component="CMB",
        )

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

    noise_freq_maps = _apply_binary_mask(manager, noise_freq_maps, unseen=True)

    if maps:
        freq_maps_plotter(
            config,
            noise_freq_maps,
            plot_dir,
            "noise_freq_maps.png",
            vmin={"I": -2, "Q": -0.5, "U": -0.5},
            vmax={"I": 2, "Q": 0.5, "U": 0.5},
        )

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
        freq_maps_plotter(
            config,
            combined_maps,
            plot_dir,
            "combined_map.png",
            vmin={"I": -300, "Q": -10, "U": -10},
            vmax={"I": 300, "Q": 10, "U": 10},
        )

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
    parser = argparse.ArgumentParser(description="Plotter for mocker output")
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
