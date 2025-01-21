import argparse
import os

import numpy as np

from megatop.utils.metadata_manager import BBmeta
from megatop.utils.preproc_utils import (
    apply_binary_mask,
    common_beam_and_nside,
    read_input_maps,
)


def preprocess_map(meta, binary_mask=True):
    meta.timer.start("preproc")
    input_maps = read_input_maps(meta)
    meta.logger.info(
        f"Input maps have shapes: {(*[input_maps[i].shape for i in range(len(meta.frequencies))],)}"
    )
    if np.all(
        np.array(meta.pre_proc_pars["common_beam_correction"]) == np.array(meta.beams_FWHM_arcmin)
    ):
        meta.logger.info(
            "Common beam correction is the same as the input beam, no need to apply it."
        )
        meta.logger.warning(
            "This is mostly for testing it might not actually represent the real noise"
        )
        freqs_maps_convolved = input_maps.astype("float64")
    else:
        freqs_maps_convolved = common_beam_and_nside(meta, input_maps)
    meta.logger.info(f"Pre-processed maps have shape: {freqs_maps_convolved.shape}")
    if binary_mask:
        freqs_maps_convolved_masked = apply_binary_mask(meta, freqs_maps_convolved)
    else:
        freqs_maps_convolved_masked = None
    meta.timer.stop("preproc", meta.logger, "Pre-processing input maps")
    return freqs_maps_convolved, freqs_maps_convolved_masked


def save_preprocessed_maps(meta, freq_maps):
    """ """
    fname = os.path.join(meta.pre_process_directory, "freqs_maps_preprocessed.npy")
    np.save(fname, freq_maps)
    meta.logger.info(f"Pre-processed maps saved to {fname}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="simplistic simulator")  # TODO change name ?
    parser.add_argument("--globals", type=str, help="Path to yaml with global parameters")

    args = parser.parse_args()
    meta = BBmeta(args.globals)
    _, freqs_maps_convolved_masked = preprocess_map(meta)
    save_preprocessed_maps(meta, freqs_maps_convolved_masked)
