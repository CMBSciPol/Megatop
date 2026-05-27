import argparse
from pathlib import Path

from megatop import Config, DataManager
from megatop.utils.mpi import get_world
from megatop.utils.precomputations import megabuster_precomputations


def main():
    parser = argparse.ArgumentParser(description="Precomputation for JAX compsep")
    parser.add_argument("--config", type=Path, required=True, help="config file")

    args = parser.parse_args()
    config = Config.load_yaml(args.config)
    manager = DataManager(config)

    world, rank, size = get_world()
    print(f"Rank {rank} of {size} is running")
    if rank == 0:
        manager.dump_config()

    if config.parametric_sep_pars.use_megabuster:
        megabuster_precomputations(manager=manager, config=config)
    else:
        print("Skipping Megabuster precomputations as use_megabuster is set to False.")


if __name__ == "__main__":
    main()
