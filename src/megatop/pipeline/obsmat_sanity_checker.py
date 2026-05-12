import argparse
from pathlib import Path

import healpy as hp
import numpy as np
import scipy as sp

from megatop import Config, DataManager
from megatop.utils import logger
from megatop.utils.binning import (
    compare_obsmat_vs_mask,
)
from megatop.utils.mpi import get_world


def obsmat_sanity_check(manager=DataManager, config=Config):
    binary_mask = hp.read_map(manager.path_to_binary_mask)
    freq_list_checks_TQU = []
    for map_set, fname in zip(config.map_sets, manager.get_obsmat_filenames(), strict=False):
        logger.info(f"Loading obsmat for {map_set.name} from {fname}")
        obsmat = sp.sparse.load_npz(fname)
        freq_list_checks_TQU.append(compare_obsmat_vs_mask(obsmat, binary_mask))
    freq_list_checks_TQU = np.array(freq_list_checks_TQU)

    assert_message = "There is a discrepancy between the binary mask and the observation matrix for the following matrix:\n"
    for i, (map_set, fname) in enumerate(
        zip(config.map_sets, manager.get_obsmat_filenames(), strict=False)
    ):
        if not freq_list_checks_TQU[i]:
            assert_message += f"{fname.name} for map set: {map_set.name} \n"

    if not np.all(freq_list_checks_TQU):
        logger.error(
            "==========================================================================================================================="
        )
        logger.error(
            "=                                                                                                                         ="
        )
        logger.error(
            "=                                                                                                                         ="
        )
        logger.error(
            "=                  Mask and Obsmat are incompatible this will cause issue in the component separation!!!                  ="
        )
        logger.error(
            "=                                                                                                                         ="
        )
        logger.error(
            "=                                                                                                                         ="
        )
        logger.error(
            "==========================================================================================================================="
        )
    else:
        logger.info(
            "==========================================================================================================================="
        )
        logger.info(
            "=                                                                                                                         ="
        )
        logger.info(
            "=                                        Mask and Obsmat sanity check successful!                                         ="
        )
        logger.info(
            "=                                                                                                                         ="
        )
        logger.info(
            "==========================================================================================================================="
        )

    assert np.all(freq_list_checks_TQU), assert_message

    return


def main():
    parser = argparse.ArgumentParser(description="Cl to r estmation")
    parser.add_argument("--config", type=Path, required=True, help="config file")

    args = parser.parse_args()
    config = Config.load_yaml(args.config)
    manager = DataManager(config)

    world, rank, size = get_world()
    logger.info(f"Rank {rank} of {size} is running")
    if rank == 0:
        manager.dump_config()

    # TODO: parallelise in the same way as the mocker
    if config.parametric_sep_pars.use_megabuster:
        # TODO: is it also an issue for just the mocker?
        obsmat_sanity_check(manager=manager, config=config)
    else:
        logger.info(
            "Megabuster is not used, skipping obsmat sanity check as it is only relevant for the component separation step."
        )


if __name__ == "__main__":
    main()
