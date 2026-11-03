import argparse
import copy

# from mpi4py.futures import MPICommExecutor
import subprocess
from pathlib import Path

import healpy as hp
import numpy as np
import pymaster as nmt
import soopercool
import yaml

from megatop import Config, DataManager
from megatop.utils import logger
from megatop.utils.binning import load_nmt_binning
from megatop.utils.mpi import get_world
from megatop.utils.TF_utils import (
    estimate_T_ell_em,
    get_reduced_TF,
    get_tf_alm_correction_fora_all_channels,
    rebuilt_from_reduced_conj,
)


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
            "lmax": config.lmax,  # 2 * config.nside + 1,
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
                        "kspace_tag": None,
                        "hits_tag": None,
                    }
                },
                "transfer_settings": {
                    "tf_ordering": "MT",  # TM is the BBmaster paper ordering. Alternative is "MT""
                    "power_law_c_ell": str(manager.get_true_spectra_TFsims_filename),
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
            logger.info(
                f"Writing soopercool config for {map_set.name} to {full_soopercool_config_path}"
            )
            yaml.dump(soopercool_config, f)
    soopercool_path = Path(soopercool.__path__[0]).parent

    logger.info("Computing Pseudo-Cell for each map sets")
    for path_soop_config in full_soopercool_config_path_list:
        # launching soopercool tf_settings
        pseudo_cell_script = soopercool_path / Path(
            "pipeline/transfer/compute_pseudo_cells_tf_estimation.py"
        )
        logger.info(
            "srun -n 100 python "
            + str(pseudo_cell_script)
            + " --globals "
            + str(path_soop_config)
            + " --verbose"
        )
        # subprocess.run(
        #     ["python", str(pseudo_cell_script), "--globals", path_soop_config, "--verbose"],
        #     check=False,
        # )

    logger.info("Computing Mode Coupling Matrix")
    mode_coupling_matrix_script = soopercool_path / Path("pipeline/get_mode_coupling.py")
    subprocess.run(
        [
            "python",
            str(mode_coupling_matrix_script),
            "--globals",
            full_soopercool_config_path_list[0],
            "--verbose",
        ],
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


def compute_and_save_reduced_transfer_functions(manager: DataManager, config: Config):
    nmt_bins = load_nmt_binning(manager)

    inv_sqrt_tf_bin_freq, inv_sqrt_tf_freq, inv_sqrt_tf_lm_freq = (
        get_tf_alm_correction_fora_all_channels(manager, config, nmt_bins, use_new_reduced_TF=True)
    )
    neg_corner_inv_sqrt_tf_lm_freq = inv_sqrt_tf_lm_freq.copy()
    neg_corner_inv_sqrt_tf_lm_freq[:, 0, 1] = -inv_sqrt_tf_lm_freq[:, 0, 1].copy()
    neg_corner_inv_sqrt_tf_lm_freq[:, 1, 0] = -inv_sqrt_tf_lm_freq[:, 1, 0].copy()

    inv_sqrt_tf_bin_freq_OLD, inv_sqrt_tf_freq_OLD, inv_sqrt_tf_lm_freq_OLD = (
        get_tf_alm_correction_fora_all_channels(manager, config, nmt_bins, use_new_reduced_TF=False)
    )
    inv_T_ellem_freq_from_alm = estimate_transfer_alm_level(manager, config)

    # print("TESTTTTTTINNNNNNNNNNGGGGGGG")
    # inv_sqrt_tf_bin_freq, inv_sqrt_tf_freq, inv_sqrt_tf_lm_freq = get_effectiv_inv_tf_least_square(manager, config, nmt_bins)
    # inv_sqrt_tf_lm_freq = np.load('/shared_home/jost/code/notebooks/test_sims_filtering/save_inv_T_ellem_freq.npy')
    # freq_alms_convolved = np.einsum("flij,fjl->fil", inv_sqrt_tf_lm_freq, freq_alms_convolved)

    # Saving the reduced transfer functions for later use
    # TODO: save TF in a seperate function like the other outputs
    fname_TF = manager.get_path_to_preprocessed_reduced_TF()
    fname_TF.parent.mkdir(parents=True, exist_ok=True)
    print(inv_sqrt_tf_lm_freq.shape)
    np.savez(
        fname_TF,
        inv_sqrt_tf_bin_freq=inv_sqrt_tf_bin_freq,
        inv_sqrt_tf_freq=inv_sqrt_tf_freq,
        inv_sqrt_tf_lm_freq=inv_sqrt_tf_lm_freq,
        OLD_inv_sqrt_tf_bin_freq=inv_sqrt_tf_bin_freq_OLD,
        OLD_inv_sqrt_tf_freq=inv_sqrt_tf_freq_OLD,
        OLD_inv_sqrt_tf_lm_freq=inv_sqrt_tf_lm_freq_OLD,
        inv_T_ellem_freq_from_alm=inv_T_ellem_freq_from_alm,
        neg_corner_inv_sqrt_tf_lm_freq=neg_corner_inv_sqrt_tf_lm_freq,
    )
    logger.info(
        f"Saved reduced transfer functions to {manager.get_path_to_preprocessed_reduced_TF()}"
    )


def reconstruct_transfer_function_from_reduced(manager: DataManager, config: Config):
    path_preprocessed_reduced_TF = manager.get_path_to_preprocessed_reduced_TF()
    logger.info(f"Loading preproc transfer function from {path_preprocessed_reduced_TF}")
    reduced_TF_from_preproc = np.load(
        path_preprocessed_reduced_TF,
        allow_pickle=True,
    )
    inv_tf_old_reduce_freq = []
    inv_tf_new_reduce_freq = []
    standard_inv_freq = []
    for f, tf_path in enumerate(manager.get_TF_filenames()):
        assert tf_path is not None, (
            f"Transfer function for frequency {config.frequencies[f]} not provided."
        )
        logger.info(f"Loading transfer function from {tf_path}")
        transfer = np.load(tf_path, allow_pickle=True)["full_tf"]
        inv_tf_old_reduce = get_reduced_TF(transfer)

        inv_tf_new_reduce = np.abs(
            rebuilt_from_reduced_conj(reduced_TF_from_preproc["inv_sqrt_tf_bin_freq"][f]).T
        )

        standard_inv = np.linalg.inv(transfer.transpose(2, 0, 1))[..., -4:, -4:]

        inv_tf_old_reduce_freq.append(inv_tf_old_reduce)
        inv_tf_new_reduce_freq.append(inv_tf_new_reduce)
        standard_inv_freq.append(standard_inv)
    standard_inv_freq = np.array(standard_inv_freq)
    inv_tf_old_reduce_freq = np.array(inv_tf_old_reduce_freq)
    inv_tf_new_reduce_freq = np.array(inv_tf_new_reduce_freq)
    logger.info(
        f"Saved reconstructed transfer functions from reduced TF to {manager.get_path_to_reconstructed_inv_TF()}"
    )
    np.savez(
        manager.get_path_to_reconstructed_inv_TF(),
        inv_tf_old_reduce_freq=inv_tf_old_reduce_freq,
        inv_tf_new_reduce_freq=inv_tf_new_reduce_freq,
        standard_inv_freq=standard_inv_freq,
    )


def estimate_transfer_alm_level(manager: DataManager, config: Config):
    inv_T_ellem_freq = []
    purify_E_global = False
    purify_B_global = True
    analysis_mask = hp.read_map(manager.path_to_analysis_mask)

    for map_set in config.map_sets:
        logger.info(f"Estimating transfer function at alm level for {map_set.name}")

        alm_E_filtered_list = []
        alm_B_filtered_list = []
        alm_E_unfiltered_list = []
        alm_B_unfiltered_list = []
        for i in range(config.map_sim_pars.TF_n_sim):
            tf_sim_path = manager.get_path_to_TF_sims_sub(id_sim=i)

            fname_unfiltered_pure_E = Path(map_set.simforTF_map_filename[1] + "_unfiltered.fits")
            fname_filtered_pure_E = Path(map_set.simforTF_map_filename[1] + "_filtered.fits")

            fname_unfiltered_pure_B = Path(map_set.simforTF_map_filename[2] + "_unfiltered.fits")
            fname_filtered_pure_B = Path(map_set.simforTF_map_filename[2] + "_filtered.fits")

            map_E_unfiltered = hp.read_map(tf_sim_path / fname_unfiltered_pure_E, field=None)
            map_E_filtered = hp.read_map(tf_sim_path / fname_filtered_pure_E, field=None)

            map_B_unfiltered = hp.read_map(tf_sim_path / fname_unfiltered_pure_B, field=None)
            map_B_filtered = hp.read_map(tf_sim_path / fname_filtered_pure_B, field=None)

            fields_E_unfiltered = nmt.NmtField(
                analysis_mask,
                map_E_unfiltered[1:],
                beam=None,
                purify_e=purify_E_global,
                purify_b=purify_B_global,
                n_iter=10,
                lmax=config.lmax,
                lmax_mask=config.lmax,
            )
            alm_E_unfiltered_list.append(fields_E_unfiltered.alm)

            fields_B_unfiltered = nmt.NmtField(
                analysis_mask,
                map_B_unfiltered[1:],
                beam=None,
                purify_e=purify_E_global,
                purify_b=purify_B_global,
                n_iter=10,
                lmax=config.lmax,
                lmax_mask=config.lmax,
            )
            alm_B_unfiltered_list.append(fields_B_unfiltered.alm)

            fields_E_filtered = nmt.NmtField(
                analysis_mask,
                map_E_filtered[1:],
                beam=None,
                purify_e=purify_E_global,
                purify_b=purify_B_global,
                n_iter=10,
                lmax=config.lmax,
                lmax_mask=config.lmax,
            )
            alm_E_filtered_list.append(fields_E_filtered.alm)

            fields_B_filtered = nmt.NmtField(
                analysis_mask,
                map_B_filtered[1:],
                beam=None,
                purify_e=purify_E_global,
                purify_b=purify_B_global,
                n_iter=10,
                lmax=config.lmax,
                lmax_mask=config.lmax,
            )
            alm_B_filtered_list.append(fields_B_filtered.alm)

        alm_E_unfiltered_list = np.array(alm_E_unfiltered_list)
        alm_B_unfiltered_list = np.array(alm_B_unfiltered_list)

        alm_E_filtered_list = np.array(alm_E_filtered_list)
        alm_B_filtered_list = np.array(alm_B_filtered_list)

        inv_T_ellem_freq.append(
            estimate_T_ell_em(
                alm_E_filtered_list[:, 0],
                alm_B_filtered_list[:, 1],
                alm_E_unfiltered_list[:],
                alm_B_unfiltered_list[:],
            )
        )
    inv_T_ellem_freq = np.array(inv_T_ellem_freq)

    inv_T_ellem_freq_shuffle1 = inv_T_ellem_freq.swapaxes(1, 2)
    # inv_T_ellem_freq_shuffle2 = inv_T_ellem_freq_shuffle1.swapaxes(2, 3)
    return inv_T_ellem_freq_shuffle1.swapaxes(2, 3)

    # logger.info(
    #     f"Saved estimated inv TF_lm to {manager.get_path_to_estimated_inv_TFlm()}"
    # )
    # np.save(manager.get_path_to_estimated_inv_TFlm(), inv_T_ellem_freq_shuffle2)


def main():
    parser = argparse.ArgumentParser(description="Cl to r estmation")
    parser.add_argument("--config", type=Path, required=True, help="config file")

    args = parser.parse_args()
    config = Config.load_yaml(args.config)
    manager = DataManager(config)

    world, rank, size = get_world()
    if rank == 0:
        manager.dump_config()
        manager.create_output_dirs(config.map_sim_pars.n_sim, config.noise_sim_pars.n_sim)

    if config.map_sim_pars.generate_sims_for_TF and config.pre_proc_pars.correct_for_TF:
        create_and_run_soopercool_yaml(manager=manager, config=config)

        compute_and_save_reduced_transfer_functions(manager=manager, config=config)
        reconstruct_transfer_function_from_reduced(manager=manager, config=config)

    else:
        logger.info(
            "Skipping transfer function computation, as generate_sims_for_TF or correct_for_TF is not set to True."
        )


if __name__ == "__main__":
    main()
