import argparse
from pathlib import Path

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np

from megatop import Config, DataManager
from megatop.utils import Timer, logger
from megatop.utils.mask import apply_binary_mask
from megatop.utils.plot import freq_maps_plotter


def plot_compsep(manager: DataManager, config: Config, id_sim: int | None = None):
    plot_dir = manager.path_to_components_plots
    plot_dir.mkdir(parents=True, exist_ok=True)

    fname_compmaps = manager.get_path_to_components_maps(id_sim)
    comp_maps = np.load(fname_compmaps)

    binary_mask = hp.read_map(manager.path_to_binary_mask)
    comp_maps = apply_binary_mask(comp_maps, binary_mask, unseen=True)

    freq_maps_plotter(
        config,
        np.array([comp_maps[0]]),
        plot_dir,
        "CMB_post_compsep_maps",
        component="CMB post-compsep",
    )
    freq_maps_plotter(
        config,
        np.array([comp_maps[1]]),
        plot_dir,
        "dust_post_compsep_maps",
        component="Dust post-compsep",
    )
    if config.parametric_sep_pars.include_synchrotron:
        freq_maps_plotter(
            config,
            np.array([comp_maps[2]]),
            plot_dir,
            "synch_post_compsep_maps",
            component="Synch post-compsep",
        )


def plot_compsep_stats(manager: DataManager, config: Config):
    if config.map_sim_pars.n_sim == 0 or config.map_sim_pars.n_sim is None:
        logger.info("No sky simulations, skipping component separation statistics plotter.")
        return

    compsep_results_params = []
    convergence_count = 0
    for sky_sims_id in range(config.map_sim_pars.n_sim):
        fname_compsepresults = manager.get_path_to_compsep_results(sub=sky_sims_id)
        compsep_results = np.load(fname_compsepresults, allow_pickle=True)
        params = compsep_results["x"]
        convergence = compsep_results["success"].astype(bool)
        if convergence:
            compsep_results_params.append(params)
            convergence_count += 1
    compsep_results_params = np.array(compsep_results_params)
    logger.info(f"Component sepatation converged successfully for of {100 * convergence_count / config.map_sim_pars.n_sim:.2f}% the maps.")

    plot_dir = manager.path_to_components_plots
    compsep_results_last = np.load(fname_compsepresults, allow_pickle=True)

    # Plotting histograms of result parameters:
    fig, axes = plt.subplots(1, compsep_results_last["params"].shape[0], figsize=(12, 5))
    axes = np.atleast_1d(axes)
    
    label_map = {
    "Dust.beta_d": r"$\beta_{dust}$",
    "Synchrotron.beta_pl": r"$\beta_{sync}$",
    }
    
    for i, (ax, param_name) in enumerate(zip(axes, compsep_results_last["params"], strict=False)):
        data = compsep_results_params[:, i]
    
        ax.hist(data, bins=25, histtype='step', density=False, color="darkblue")
    
        mean_param = np.mean(data)
        std_param = np.std(data)
    
        ax.axvline(mean_param, color="mediumvioletred", linestyle="-", linewidth=1.5)
    
        ax.grid(True, linestyle="--", color="lightgrey", alpha=0.7)
        ax.set_xlabel(label_map.get(param_name, param_name))
        ax.set_ylabel("Counts")
        title = fr"${label_map.get(param_name, param_name).strip('$')} = {mean_param:.3f} \pm {std_param:.5f}$"
        ax.set_title(title)

    plt.savefig(plot_dir / Path("statistics_compsep.png"))  # , bbox_inches='tight')
    plt.close()
    return


def main():
    parser = argparse.ArgumentParser(description="Plotter for component separation output")
    parser.add_argument("--config", type=Path, help="config file")
    args = parser.parse_args()
    if args.config is None:
        logger.warning("No config file provided, using example config")
        config = Config.get_example()
    else:
        config = Config.load_yaml(args.config)
    manager = DataManager(config)
    manager.dump_config()

    n_sim_sky = config.map_sim_pars.n_sim
    if n_sim_sky == 0:
        id_sim = None
    else:
        logger.info("Plotting only simulation #0")
        id_sim = 0

    logger.info("Plotting comp sep outputs...")
    with Timer("comp-sep-plotter"):
        plot_compsep(manager, config, id_sim=id_sim)
        plot_compsep_stats(manager, config)


if __name__ == "__main__":
    main()
