import argparse
import multiprocessing as mp
from pathlib import Path

import healpy as hp
import numpy as np

from megatop import Config, DataManager
from megatop.utils import Timer, logger
from megatop.utils.mask import apply_binary_mask
from megatop.utils.preproc import common_beam_and_nside, read_input_maps


def preprocess_map(manager: DataManager, config: Config, mask_output=True):
    input_maps = read_input_maps(manager.get_maps_filenames())
    logger.info(
        f"Input maps have shapes: {[input_maps[i].shape for i in range(len(config.frequencies))]}"
    )
    if np.all(np.array(config.pre_proc_pars.common_beam_correction) == np.array(config.beams)):
        logger.info("Common beam correction is the same as the input beam, no need to apply it.")
        logger.warning("This is mostly for testing it might not actually represent the real noise")
        freq_maps_convolved = input_maps.astype("float64")
    else:
        freq_maps_convolved = common_beam_and_nside(
            nside=config.nside,
            common_beam=config.pre_proc_pars.common_beam_correction,
            frequency_beams=config.beams,
            freq_maps=input_maps,
        )
    logger.info(f"Pre-processed maps have shape: {freq_maps_convolved.shape}")

    if mask_output:
        binary_mask = hp.read_map(manager.path_to_binary_mask)
        freq_maps_convolved = apply_binary_mask(freq_maps_convolved, binary_mask=binary_mask)
    return freq_maps_convolved


def save_preprocessed_maps(manager: DataManager, freq_maps):
    manager.path_to_preproc.mkdir(exist_ok=True, parents=True)
    fname = manager.get_path_to_preprocessed_maps()
    logger.info(f"Saving pre-processed maps to {fname}")
    np.save(fname, freq_maps)


def preproc_and_save(args, id_sim=None):
    if id_sim is None:  # Running only one simulation
        if args.config is None:
            logger.warning("No config file provided, using example config")
            config = Config.get_example()
        else:
            config = Config.from_yaml(args.config)
    else:
        if not args.config_root:
            logger.warning("No config root provided, required for multiple simulations. exiting")
            raise AttributeError
        fname_config = args.config_root.with_name(f"{args.config_root.name}_{id_sim:04d}.yml")
        config = Config.from_yaml(fname_config)
    manager = DataManager(config)
    manager.dump_config()
    with Timer("preprocesser"):
        freq_maps_convolved = preprocess_map(manager, config, mask_output=True)
    save_preprocessed_maps(manager, freq_maps_convolved)


def main():
    parser = argparse.ArgumentParser(description="Preprocesser")
    parser.add_argument("--config", type=Path, help="config file")
    parser.add_argument(
        "--config_root", type=Path, help="config file root (will be appended  by {id_sim:04d})"
    )
    parser.add_argument("--Nsims", type=int, help="Number of simulations performed")
    parser.add_argument(
        "--noise-only", action="store_true", help="generate noise-only sims and save them to disk"
    )
    parser.add_argument(
        "--nomultiproc", action="store_true", help="don't use multprocessing parallelisation"
    )
    args = parser.parse_args()

    if args.config and not args.noise_only and not args.Nsims:  # Prioritize --config if provided
        preproc_and_save(args)
        return

    if args.config_root:  # Multiple simulations mode
        if not args.Nsims:
            logger.warning("Nsims not specified, will only run one ")
            Nsims = 1
        else:
            Nsims = args.Nsims

        num_workers = 1 if args.nomultiproc else min(mp.cpu_count(), Nsims)
        logger.info(f"Using {num_workers} worker processes")
        if num_workers > 1:
            mp.set_start_method("spawn", force=True)  # Ensure a clean multiprocessing start
            with mp.Pool(num_workers) as pool:
                pool.starmap(
                    preproc_and_save,
                    [(args, id_sim) for id_sim in range(Nsims)],
                )
        else:
            for id_sim in range(Nsims):
                preproc_and_save(args, id_sim)
    else:
        # Default case: no arguments provided, run single simulation with example config
        preproc_and_save(args)


if __name__ == "__main__":
    main()
