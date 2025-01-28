import argparse
from pathlib import Path

import numpy as np

from megatop import Config
from megatop.utils import Timer, logger
from megatop.utils.preproc import _apply_binary_mask, _common_beam_and_nside, _read_input_maps


def preprocess_map(config: Config, binary_mask=True):
    timer = Timer()
    timer.start("preproc")
    input_maps = _read_input_maps(config)
    logger.info(
        f"Input maps have shapes: {[input_maps[i].shape for i, _ in enumerate(config.frequencies)]}"
    )
    if np.all(np.array(config.pre_proc_pars.common_beam_correction) == np.array(config.beams)):
        logger.info("Common beam correction is the same as the input beam, no need to apply it.")
        logger.warning("This is mostly for testing it might not actually represent the real noise")
        freq_maps_convolved = input_maps.astype("float64")
    else:
        freq_maps_convolved = _common_beam_and_nside(config, input_maps)
    logger.info(f"Pre-processed maps have shape: {freq_maps_convolved.shape}")
    if binary_mask:
        freq_maps_convolved_masked = _apply_binary_mask(config, freq_maps_convolved)
    else:
        freq_maps_convolved_masked = None
    timer.stop("preproc", "Pre-processing input maps")
    return freq_maps_convolved, freq_maps_convolved_masked


def save_preprocessed_maps(config: Config, freq_maps):
    # TODO: should this name be a parameter in the config?
    # TODO: why use npy format?
    config.path_to_preproc.mkdir(exist_ok=True, parents=True)
    fname = config.path_to_preproc / "freq_maps_preprocessed.npy"
    np.save(fname, freq_maps)
    logger.info(f"Pre-processed maps saved to {fname}")


def main():
    parser = argparse.ArgumentParser(description="Preprocesser")
    parser.add_argument("--config", type=Path, help="config file")
    args = parser.parse_args()
    config = Config.from_yaml(args.config)
    config.dump()
    _, freq_maps_convolved_masked = preprocess_map(config)
    save_preprocessed_maps(config, freq_maps_convolved_masked)


if __name__ == "__main__":
    main()
