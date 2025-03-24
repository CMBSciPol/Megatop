import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from megatop import Config, DataManager
from megatop.utils import logger


def plot_r_statistics(managers, configs):
    """Plots r statistics for different pipeline configurations in a single figure, grouping by sky model."""
    sky_model_dict = defaultdict(list)  # Group data by sky model

    for manager, config in zip(managers, configs, strict=False):
        n_sim_sky = config.map_sim_pars.n_sim
        sky_model = "".join(config.map_sim_pars.sky_model)

        r_means = []
        r_stds = []

        for id_sim in range(n_sim_sky):
            fname_chains = manager.get_path_to_mcmc_chains(sub=id_sim)
            mcmc = np.load(fname_chains, allow_pickle=True)
            chains = mcmc["mcmc_chains"]
            param_names = mcmc["param_names"]

            idx_r = np.where(param_names == "r")[0][0]
            r_samples = chains[:, idx_r]

            r_means.append(np.mean(r_samples))
            r_stds.append(np.std(r_samples))

        r_mean_final = np.mean(r_means)  # Mean of means
        r_std_final = np.mean(r_stds)  # Mean of standard deviations

        if config.cl2r_pars.dust_marg:
            color = "mediumvioletred"
            label = r"$\theta = (r, A_{\rm lens}, A_{\rm dust})$"
        else:
            color = "darkblue"
            label = r"$\theta = (r, A_{\rm lens})$"

        sky_model_dict[sky_model].append((r_mean_final * 1e3, r_std_final * 1e3, color, label))

    sky_models = list(sky_model_dict.keys())
    x_positions = np.arange(len(sky_models))

    plt.figure(figsize=(8, 6))
    offset = 0.1
    for i, sky_model in enumerate(sky_models):
        entries = sky_model_dict[sky_model]
        for j, (mean, std, color, label) in enumerate(entries):
            plt.errorbar(
                x_positions[i] + j * offset,
                mean,
                yerr=std,
                fmt="o",
                color=color,
                capsize=5,
                label=label if i == 0 else "_nolegend_",
            )

    plt.axhline(0, color="darkgrey", linestyle="--", linewidth=0.75, alpha=0.6)
    plt.xticks(x_positions, sky_models, rotation=45)
    plt.xlabel("sky model", fontsize=12)
    plt.ylabel(r"$(r \pm \sigma(r)) \times 10^3$", fontsize=12)

    handles, labels = plt.gca().get_legend_handles_labels()
    unique_legend = dict(zip(labels, handles, strict=False))
    plt.legend(unique_legend.values(), unique_legend.keys())
    plt.tight_layout()

    plot_dir = manager.path_to_mcmc_plots
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_name = "r_stats_model_comparison"
    plt.savefig(plot_dir / plot_name, bbox_inches="tight")

    plt.show()
    plt.clf()


def main():
    parser = argparse.ArgumentParser(description="Plotter for component separation output")
    parser.add_argument(
        "--configs", type=Path, nargs="+", help="List of config files"
    )  # Accept multiple configs
    args = parser.parse_args()

    if not args.configs:
        logger.warning("No config file provided, using example config")
        configs = [Config.get_example()]
    else:
        configs = [Config.load_yaml(cfg) for cfg in args.configs]

    managers = [DataManager(cfg) for cfg in configs]

    for manager in managers:
        manager.dump_config()

    plot_r_statistics(managers, configs)


if __name__ == "__main__":
    main()
