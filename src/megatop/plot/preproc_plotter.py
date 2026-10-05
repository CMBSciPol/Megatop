import argparse
import os
from pathlib import Path

import healpy as hp
import numpy as np

import megatop.utils.harmonic as hu
from megatop import Config, DataManager
from megatop.utils import Timer, logger
from megatop.utils.mask import apply_binary_mask
from megatop.utils.plot import freq_maps_plotter, group_indices_by_experiment, plotTTEEBB

HEALPY_DATA_PATH = os.getenv("HEALPY_LOCAL_DATA", None)


def plot_preprocessed_maps(manager, config, id_sim=None, maps=True, cls=True):
    plot_dir = manager.path_to_preproc_plots
    plot_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Plotting pre-processing outputs")

    with Timer("load-freq-maps"):
        preproc_maps_fname = manager.get_path_to_preprocessed_maps(id_sim)
        logger.debug(f"Loading input maps from {preproc_maps_fname}")
        freq_maps_preprocessed = np.load(preproc_maps_fname)

        experiments = set(m.exp_tag for m in config.map_sets)
        binary_mask = {exp: hp.read_map(manager.path_to_binary_mask(exp)) for exp in experiments}

        for i_m, map_set in enumerate(config.map_sets):
            _ = apply_binary_mask(
                freq_maps_preprocessed[i_m], binary_mask=binary_mask[map_set.exp_tag], unseen=True
            )

    if cls:  # plotting the spectra
        lmax = config.plot_pars.lmax_plot
        spectra_array = np.array(
            [hu.anafast(freq_maps_preprocessed[i], lmax=lmax) for i in range(len(config.map_sets))]
        )

    for exp, idx in group_indices_by_experiment(config.map_sets).items():
        exp_config = config.model_copy(update={"map_sets": [config.map_sets[i] for i in idx]})

        if maps:  # Plotting the maps
            freq_maps_plotter(
                exp_config, freq_maps_preprocessed[idx], plot_dir, f"pre_processed_maps_{exp}"
            )

        if cls:
            plotTTEEBB(
                plot_dir=plot_dir,
                freqs=exp_config.frequencies,
                Cl=spectra_array[idx],
                save_name=f"spectra_pre_processed_anafast_{exp}",
                y_axis_label=r"$C_\ell$ pre-processed",
                use_D_ell=False,
                lims_x=None,
                lims_y=None,
            )


def main():
    parser = argparse.ArgumentParser(description="Plotter for preprocessing output")
    parser.add_argument("--config", type=Path, help="config file")
    args = parser.parse_args()
    if args.config is None:
        logger.warning("No config file provided, using example config")
        config = Config.get_example()
    else:
        config = Config.load_yaml(args.config)
    manager = DataManager(config)
    manager.dump_config()

    logger.info("Plotting preprocessing outputs...")

    n_sim_sky = config.map_sim_pars.n_sim
    if n_sim_sky == 0:
        id_sim = None
    else:
        logger.info("Plotting only simulation #0")
        id_sim = 0

    with Timer("preproc-plotter"):
        plot_preprocessed_maps(manager, config, id_sim=id_sim)


if __name__ == "__main__":
    main()