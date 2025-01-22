import argparse
import os

import healpy as hp
import matplotlib.pyplot as plt
from matplotlib import cm

from megatop.utils import BBmeta
from megatop.utils.mask import get_spin_derivatives


def plotter(meta):
    cmap = cm.RdBu
    cmap.set_under("w")
    plot_dir = meta.plot_dir_from_output_dir(meta.mask_directory_rel)

    # Plotting hits map
    nhits = meta.read_hitmap()
    plt.figure(figsize=(16, 9))
    hp.mollview(nhits, cmap=cmap, cbar=True, title="Hits map")
    hp.graticule()
    plt.savefig(os.path.join(plot_dir, meta.masks["nhits_map"]).replace(".fits", ".png"))
    plt.clf()

    # Plotting binary mask map
    binary_mask = meta.read_mask("binary")
    plt.figure(figsize=(16, 9))
    hp.mollview(binary_mask, cmap=cmap, cbar=True, title="Binary mask, derived from hits map")
    hp.graticule()
    plt.savefig(os.path.join(plot_dir, meta.masks["binary_mask"]).replace(".fits", ".png"))
    plt.clf()

    # Plotting galactic mask
    if "galactic" in meta.masks["include_in_mask"]:
        galactic_mask = meta.read_mask("galactic")
        plt.figure(figsize=(16, 9))
        hp.mollview(
            galactic_mask,
            cmap=cmap,
            cbar=True,
            title=f"Galactic map {meta.masks['galactic_mask_mode']}",
        )
        hp.graticule()
        plt.savefig(
            os.path.join(
                plot_dir,
                f"{meta.masks['galactic_mask_root']}_{meta.masks['galactic_mask_mode']}.png",
            )
        )
        plt.clf()

    # Plotting point source mask
    if "point_source" in meta.masks["include_in_mask"]:
        point_source_mask = meta.read_mask("point_source")
        plt.figure(figsize=(16, 9))
        hp.mollview(point_source_mask, cmap=cmap, cbar=True, title="Point source mask")
        hp.graticule()
        plt.savefig(
            os.path.join(plot_dir, meta.masks["point_source_mask"]).replace(".fits", ".png")
        )
        plt.clf()

    # Plotting final analysis mask
    final_mask = meta.read_mask("analysis")
    plt.figure(figsize=(16, 9))
    hp.mollview(final_mask, cmap=cmap, cbar=True, title="Final analysis mask")
    hp.graticule()
    plt.savefig(os.path.join(plot_dir, meta.masks["analysis_mask"]).replace(".fits", ".png"))
    plt.clf()

    first, second = get_spin_derivatives(final_mask)
    # Plot first spin derivative of analysis mask
    plt.figure(figsize=(16, 9))
    hp.mollview(
        first, title="First spin derivative of the final analysis mask", cmap=cmap, cbar=True
    )
    hp.graticule()
    plt.savefig(os.path.join(plot_dir, meta.masks["analysis_mask"]).replace(".fits", "_first.png"))
    plt.clf()

    # Plot second spin derivative of analysis mask
    plt.figure(figsize=(16, 9))
    hp.mollview(
        second, title="Second spin derivative of the final analysis mask", cmap=cmap, cbar=True
    )
    hp.graticule()
    plt.savefig(os.path.join(plot_dir, meta.masks["analysis_mask"]).replace(".fits", "_second.png"))
    plt.clf()


def main():
    parser = argparse.ArgumentParser(description="simplistic simulator")
    parser.add_argument("--globals", type=str, help="Path to yaml with global parameters")
    args = parser.parse_args()
    meta = BBmeta(args.globals)
    plotter(meta)


if __name__ == "__main__":
    main()
