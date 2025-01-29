import argparse
from pathlib import Path

import numpy as np

from megatop import Config, DataManager
from megatop.utils import Timer, logger
from megatop.utils.preproc import _apply_binary_mask, _common_beam_and_nside, _read_input_maps


def preprocess_map(manager: DataManager, config: Config, binary_mask=True):
    timer = Timer()
    timer.start("preproc")
    input_maps = _read_input_maps(manager)
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
        freq_maps_convolved_masked = _apply_binary_mask(manager, freq_maps_convolved)
    else:
        freq_maps_convolved_masked = None
    timer.stop("preproc", "Pre-processing input maps")
    return freq_maps_convolved, freq_maps_convolved_masked


def save_preprocessed_maps(manager: DataManager, freq_maps):
    manager.path_to_preproc.mkdir(exist_ok=True, parents=True)
    fname = manager.get_path_to_preprocessed_maps()
    logger.info(f"Saving pre-processed maps to {fname}")
    np.save(fname, freq_maps)


def main():
    parser = argparse.ArgumentParser(description="Preprocesser")
    parser.add_argument("--config", type=Path, help="config file")
    args = parser.parse_args()
    if args.config is None:
        logger.warning("No config file provided, using example config")
        config = Config.get_example()
    else:
        config = Config.from_yaml(args.config)
    manager = DataManager(config)
    manager.dump_config()
    _, freq_maps_convolved_masked = preprocess_map(manager, config)
    save_preprocessed_maps(manager, freq_maps_convolved_masked)


if __name__ == "__main__":
    main()
