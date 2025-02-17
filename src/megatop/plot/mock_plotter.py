import argparse
from pathlib import Path

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np

from megatop import Config, DataManager
from megatop.utils import Timer, logger, mock
from megatop.utils.mask import apply_binary_mask
from megatop.utils.plot import freq_maps_plotter, plotTTEEBB, plotTTEEBB_diff
from megatop.utils.preproc import read_input_maps


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

    fg_freq_maps = mock.generate_map_fgs_pysm(
        config.frequencies, config.nside, config.map_sim_pars.sky_model
    )
    fg_freq_maps_beamed = np.zeros_like(fg_freq_maps)

    for i_f, _f in enumerate(config.frequencies):
        fg_freq_maps_beamed[i_f] = mock.beam_winpix_correction(
            config.nside, fg_freq_maps[i_f], config.beams[i_f]
        )
    binary_mask = hp.read_map(manager.path_to_binary_mask)

    fg_freq_maps = apply_binary_mask(fg_freq_maps, binary_mask, unseen=True)
    fg_freq_maps_beamed = apply_binary_mask(fg_freq_maps_beamed, binary_mask, unseen=True)

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
            use_D_ell=False,
            y_axis_label=r"$C_\ell$ fg unbeamed",
            lims_x=None,
            lims_y=None,
        )
        plotTTEEBB(
            plot_dir=plot_dir,
            freqs=config.frequencies,
            Cl=cls_beamed,
            save_name="fg_cls_beamed.png",
            use_D_ell=False,
            y_axis_label=r"$C_\ell$ fg beamed",
            lims_x=None,
            lims_y=None,
        )


def plot_cmb_sims(manager, config, maps=True, cls=True):
    Cl_cmb_model = mock.get_Cl_CMB_model_from_manager(manager)
    cmb_map = mock.generate_map_cmb(
        Cl_cmb_model, config.nside, fixed_cmb=config.map_sim_pars.fixed_cmb
    )

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
            axis_labels=[r"$C_\ell^{\rm CMB}$", "Relative diff"],
            use_D_ell=False,
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
        n_ell, map_white_noise_levels = mock.get_noise(config, fsky_binary)
        noise_freq_maps = mock.get_noise_map_from_white_noise(
            config.frequencies, config.nside, map_white_noise_levels
        )

    elif config.noise_sim_pars.noise_option == "noise_spectra":
        n_ell, map_white_noise_levels = mock.get_noise(config, fsky_binary)
        noise_freq_maps = mock.get_noise_map_from_noise_spectra(
            config.frequencies, config.nside, n_ell
        )

    noise_freq_maps = apply_binary_mask(noise_freq_maps, binary_mask, unseen=True)

    if config.noise_sim_pars.include_nhits:
        noise_freq_maps = mock.include_hits_noise(
            noise_freq_maps, hp.read_map(manager.path_to_nhits_map), binary_mask
        )  # TODO move reading of maps to manager

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

    combined_maps = np.array(read_input_maps(manager.get_maps_filenames()))
    binary_mask = hp.read_map(manager.path_to_binary_mask)

    combined_maps = apply_binary_mask(combined_maps, binary_mask, unseen=True)

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
            lims_x=None,
            lims_y=None,
            legend_labels=(r"combined $C_\ell$ $\nu=$",),
            y_axis_label=r"$C_\ell$ fg unbeamed",
            use_D_ell=False,
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

    with Timer("mock-plotter"):
        plot_fiducial_spectra(manager)
        plot_fg_sims(manager, config)
        plot_cmb_sims(manager, config)
        if config.noise_sim_pars.noise_option != "no_noise":
            plot_noise_sims(manager, config)
        plot_saved_sims(manager, config)


if __name__ == "__main__":
    main()
