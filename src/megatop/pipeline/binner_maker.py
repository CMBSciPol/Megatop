import pymaster as nmt
import numpy as np
import argparse
from pathlib import Path
import IPython
from megatop.utils.mpi import get_world

from megatop.utils import logger
from megatop import Config, DataManager
from megatop.utils.binning import (
    create_binning,
)


def binning_maker(manager: DataManager, config: Config):    
    bin_low, bin_high, bin_centre = create_binning(
        config.nside, config.map2cl_pars.delta_ell, end_first_bin=config.lmin
    )
    bin_index_lminlmax = np.where((bin_low >= config.lmin) & (bin_high <= config.lmax))[0]

    path = manager.path_to_binning
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        bin_low=bin_low,
        bin_high=bin_high,
        bin_centre=bin_centre,
        bin_index_lminlmax=bin_index_lminlmax,
        bin_centre_lminlmax=bin_centre[bin_index_lminlmax],
    )
    logger.info(f"Saving binning to {path}")

def main():
    parser = argparse.ArgumentParser(description="Cl to r estmation")
    parser.add_argument("--config", type=Path, required=True, help="config file")

    args = parser.parse_args()
    config = Config.load_yaml(args.config)
    manager = DataManager(config)
    
    world, rank, size = get_world()
    print(f"Rank {rank} of {size} is running")
    if rank == 0:
        manager.dump_config()

    binning_maker(manager=manager, config=config)


if __name__ == "__main__":
    main()
