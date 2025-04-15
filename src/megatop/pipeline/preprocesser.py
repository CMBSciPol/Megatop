import argparse
from functools import partial
from pathlib import Path

import healpy as hp
import numpy as np
from mpi4py.futures import MPICommExecutor

from megatop import Config, DataManager
from megatop.utils import Timer, logger
from megatop.utils.mask import apply_binary_mask
from megatop.utils.preproc import common_beam_and_nside, read_input_maps


def preprocess_map(
    manager: DataManager, config: Config, id_sim: int | None = None, mask_output=True
):
    input_maps = read_input_maps(manager.get_maps_filenames(sub=id_sim))
    logger.info(
        f"Input maps have shapes: {[input_maps[i].shape for i in range(len(config.frequencies))]}"
    )
    if np.all(np.array(config.pre_proc_pars.common_beam_correction) == np.array(config.beams)):
        logger.info("Common beam correction is the same as the input beam, no need to apply it.")
        logger.warning("This is mostly for testing it might not actually represent the real noise")
        freq_maps_convolved = np.array(input_maps)
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


def save_preprocessed_maps(manager: DataManager, freq_maps, id_sim: int | None = None):
    fname = manager.get_path_to_preprocessed_maps(sub=id_sim)
    fname.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving pre-processed maps to {fname}")
    np.save(fname, freq_maps)


def preproc_and_save(config: Config, manager: DataManager, id_sim: int | None = None) -> None | int:
    with Timer("preprocesser"):
        freq_maps_convolved = preprocess_map(manager, config, id_sim=id_sim)
    save_preprocessed_maps(manager, freq_maps_convolved, id_sim=id_sim)
    return id_sim


def main():
    parser = argparse.ArgumentParser(
        description="Preprocesser", epilog="mpi4py is required to run this script"
    )
    parser.add_argument("--config", type=Path, required=True, help="config file")

    args = parser.parse_args()
    config = Config.load_yaml(args.config)
    manager = DataManager(config)

    n_sim_sky = config.map_sim_pars.n_sim
    if n_sim_sky == 0:  # No sky simulations: run preprocessing on the real data
        preproc_and_save(config, manager, id_sim=None)
    else:
        with MPICommExecutor() as executor:
            if executor is not None:
                logger.info(f"Distributing work to {executor.num_workers} workers")  # pyright: ignore[reportAttributeAccessIssue]
                func = partial(preproc_and_save, config, manager)
                for result in executor.map(func, range(n_sim_sky), unordered=True):  # pyright: ignore[reportAttributeAccessIssue]
                    logger.info(f"Finished preprocessing map {result + 1} / {n_sim_sky}")


if __name__ == "__main__":
    main()
