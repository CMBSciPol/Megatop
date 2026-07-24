import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from megatop import Config, DataManager
from megatop.utils import Timer, logger
from megatop.utils.binning import load_nmt_binning


def plot_transfer_function(manager, config):
    """
    Plot the transfer function given an input dictionary.
    Based on SOOPERCOOL plotting function.
    """
    plot_dir = manager.path_to_transfer_functions_plots
    plot_dir.mkdir(parents=True, exist_ok=True)

    nmt_bins = load_nmt_binning(manager)
    lb = nmt_bins.get_effective_ells()
    lmin = 2
    lmax = 2 * config.nside + 1

    field_pairs = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
    npan = len(field_pairs)
    plt.figure(figsize=(25 * npan / 9, 25 * npan / 9))
    grid = plt.GridSpec(npan, npan, hspace=0.3, wspace=0.3)

    # tf_dict could contain several versions of a TF to compare.
    # If TF_dict contains only a single TF, this ensures compatibility.
    for f, tf_path in enumerate(manager.get_TF_filenames()):
        if tf_path == Path():
            logger.warning(
                f"Transfer function for frequency {config.frequencies[f]} is not provided, skipping."
            )
            continue
        logger.info(f"Loading transfer function from {tf_path}")
        # Loading TF:
        tf_dict = np.load(tf_path, allow_pickle=True)
        if "TT_to_TT" in tf_dict:
            tf_dict = {"TF": tf_dict}

        for label, tf in tf_dict.items():
            for id1, f1 in enumerate(field_pairs):
                for id2, f2 in enumerate(field_pairs):
                    ax = plt.subplot(grid[id1, id2])
                    expected = 1.0 if f1 == f2 else 0.0
                    ylims = [0, 1.05] if f1 == f2 else [-0.01, 0.01]

                    ax.axhline(expected, color="k", ls="--", zorder=6)
                    # We need to understand the offdigonal TF panels in the
                    # presence of NaMaster purification - we don't have a clear
                    # interpretation.
                    ax.set_title(f"{f1} $\\rightarrow$ {f2}", fontsize=14)
                    # ax.plot(lb, tf[f"{f1}_to_{f2}"], label=label)
                    ax.plot(lb, tf[f"{f1}_to_{f2}"], label=f"{config.frequencies[f]} GHz")

                    if id1 == npan - 1:
                        ax.set_xlabel(r"$\ell$", fontsize=14)
                    else:
                        ax.set_xticks([])

                        if f1 != f2:
                            ax.ticklabel_format(
                                axis="y", style="scientific", scilimits=(0, 0), useMathText=True
                            )

                    ax.set_xlim(lmin, lmax)
                    ax.set_ylim(ylims[0], ylims[1])
                    if label is not None and id1 == 0 and id2 == npan - 1:
                        ax.legend(fontsize=10, bbox_to_anchor=[1, 1], loc="upper left")

    plt.savefig(plot_dir / "transfer_function_perfreq.png", bbox_inches="tight")
    plt.close()
    plt.clf()


def main():
    parser = argparse.ArgumentParser(description="Plotter for preprocessing output")
    parser.add_argument("--config", type=Path, help="config file")
    args = parser.parse_args()

    args = parser.parse_args()
    config = Config.load_yaml(args.config)
    manager = DataManager(config)

    manager.dump_config()

    logger.info("Plotting preprocessing outputs...")

    with Timer("preproc-plotter"):
        plot_transfer_function(manager, config)


if __name__ == "__main__":
    main()
