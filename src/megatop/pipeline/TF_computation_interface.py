import argparse
import copy

# from mpi4py.futures import MPICommExecutor
import subprocess
from pathlib import Path

import soopercool
import yaml

from megatop import Config, DataManager
from megatop.utils import logger
from megatop.utils.mpi import get_world


def deep_merge(base, override):
    """Recursively merge override into base."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def create_and_run_soopercool_yaml(manager: DataManager, config: Config):
    base_config = {
        "output_directory": str(manager.path_to_transfer_functions_parents),
        "transfer_dir": str(manager.path_to_transfer_functions_parents),
        "inputs_dir": str(manager.path_to_root),
        "masks": {
            "analysis_mask": str(manager.path_to_analysis_mask),
        },
        "general_pars": {
            "pix_type": "hp",
            "car_template": None,
            "nside": config.nside,
            "lmin": 2,
            "lmax": 2 * config.nside + 1,
            "binning_file": str(manager.path_to_binning),
            "pure_B": config.map2cl_pars.purify_b,
            "beam_floor": None,
        },
        "transfer_settings": {
            "transfer_directory": str(
                manager.path_to_transfer_functions_parents / Path("transfer_functions_output")
            ),
            "couplings_dir": str(
                manager.path_to_transfer_functions_parents / Path("couplings_output")
            ),
            "tf_est_num_sims": config.map_sim_pars.TF_n_sim,
            "sim_id_start": 0,
            "do_not_beam_est_sims": True,
            "beams_list": [],
        },
    }

    # creating specific params for each map_sets
    variable_config = []
    for map_set in config.map_sets:
        variable_config.append(
            {
                "map_sets": {
                    f"{map_set.name}": {
                        "map_dir": None,
                        "bean_dir": None,
                        "map_template": None,
                        "beam_file": None,
                        "n_bundles": 1,
                        "freq_tag": map_set.freq_tag,
                        "exp_tag": map_set.exp_tag,
                        "filtering_tag": f"{map_set.name}",
                    }
                },
                "transfer_settings": {
                    "unfiltered_map_dir": {f"{map_set.name}": str(manager.path_to_TF_sims_maps)},
                    "unfiltered_map_template": {
                        f"{map_set.name}": "{id_sim:04d}/simforTF_{pure_type}_"
                        + f"{map_set.name}_unfiltered.fits"
                    },
                    "filtered_map_dir": {f"{map_set.name}": str(manager.path_to_TF_sims_maps)},
                    "filtered_map_template": {
                        f"{map_set.name}": "{id_sim:04d}/simforTF_{pure_type}_"
                        + f"{map_set.name}_filtered.fits"
                    },
                },
            }
        )

    full_soopercool_config_path_list = []
    for var, map_set in zip(variable_config, config.map_sets, strict=True):
        soopercool_config = deep_merge(base_config, var)
        dir_soopercool_config = manager.path_to_transfer_functions_parents / Path("config_files")
        dir_soopercool_config.mkdir(parents=True, exist_ok=True)
        fname_soopercool_config = Path(f"{map_set.name}_transfer_function_config.yaml")
        full_soopercool_config_path = dir_soopercool_config / fname_soopercool_config
        full_soopercool_config_path_list.append(full_soopercool_config_path)
        with Path.open(full_soopercool_config_path, "w") as f:
            yaml.dump(soopercool_config, f)

    soopercool_path = Path(soopercool.__path__[0]).parent
    logger.info("Computing Pseudo-Cell for each map sets")
    for path_soop_config in full_soopercool_config_path_list:
        # launching soopercool tf_settings
        pseudo_cell_script = soopercool_path / Path(
            "pipeline/transfer/compute_pseudo_cells_tf_estimation.py"
        )
        subprocess.run(
            ["python", str(pseudo_cell_script), "--globals", path_soop_config, "--verbose"],
            check=False,
        )

    logger.info("Computing Transfer Functions for each map sets")
    TF_paths_checks = manager.get_TF_filenames()
    for i, path_soop_config in enumerate(full_soopercool_config_path_list):
        transfer_functions_script = soopercool_path / Path(
            "pipeline/transfer/compute_transfer_function.py"
        )
        subprocess.run(
            ["python", str(transfer_functions_script), "--globals", path_soop_config], check=False
        )
        logger.info(
            f"Transfer Functions computed for {config.map_sets[i].name} and saved to {TF_paths_checks[i]}"
        )


def main():
    parser = argparse.ArgumentParser(description="Cl to r estmation")
    parser.add_argument("--config", type=Path, required=True, help="config file")

    args = parser.parse_args()
    config = Config.load_yaml(args.config)
    manager = DataManager(config)

    world, rank, size = get_world()
    if rank == 0:
        manager.dump_config()

    if config.map_sim_pars.generate_sims_for_TF and config.pre_proc_pars.correct_for_TF:
        create_and_run_soopercool_yaml(manager=manager, config=config)
    else:
        logger.info(
            "Skipping transfer function computation, as generate_sims_for_TF or correct_for_TF is not set to True."
        )

    # TODO: parallelize
    # n_sim_sky = config.map_sim_pars.n_sim
    # if n_sim_sky == 0:
    #     create_and_run_soopercool_yaml(manager=manager, config=config)
    # else:
    #     with MPICommExecutor() as executor:
    #         if executor is not None:
    #             logger.info(f"Distributing work to {executor.num_workers} workers")
    #             func = partial(create_and_run_soopercool_yaml, manager, config)
    #             for result in executor.map(func, range(n_sim_sky), unordered=True):
    #                 logger.info(f"Finished mcmc run on map {result + 1} / {n_sim_sky}")


if __name__ == "__main__":
    main()
