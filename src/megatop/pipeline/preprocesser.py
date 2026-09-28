import argparse
from functools import partial
from pathlib import Path

import healpy as hp
import numpy as np

from megatop import Config, DataManager
from megatop.utils import Timer, logger
from megatop.utils.mask import apply_binary_mask
from megatop.utils.mpi import get_world
from megatop.utils.preproc import alm_common_beam, common_beam_and_nside, read_input_maps


def preprocess_map(
    manager: DataManager, config: Config, id_sim: int | None = None, mask_output=True
):
    input_maps = read_input_maps(manager.get_maps_filenames(id_sim))
    logger.info(
        f"Input maps have shapes: {[input_maps[i].shape for i in range(len(config.frequencies))]}"
    )

    use_harmonic = config.parametric_sep_pars.use_harmonic_compsep

    freq_maps_convolved = common_beam_and_nside(
        nside=config.nside,
        common_beam=config.pre_proc_pars.common_beam_correction,
        frequency_beams=config.beams,
        freq_maps=input_maps,
        lmax=config.lmax,
    )
    logger.info(f"Pre-processed maps have shape: {freq_maps_convolved.shape}")

    if not use_harmonic:
        if mask_output:
            binary_mask = hp.read_map(manager.path_to_binary_mask)
            freq_maps_convolved = apply_binary_mask(freq_maps_convolved, binary_mask=binary_mask)
        return freq_maps_convolved

    logger.info("Using harmonic pipeline for component separation. Pre-processing will output alms")
    analysis_mask = hp.read_map(manager.path_to_analysis_mask)

    freq_alms_convolved = alm_common_beam(
        common_beam=config.pre_proc_pars.common_beam_correction,
        frequency_beams=config.beams,
        freq_maps=np.array(input_maps),
        lmax=config.lmax,
        analysis_mask=analysis_mask,
        purify_e=config.map2cl_pars.purify_e,
        purify_b=config.map2cl_pars.purify_b,
        use_namaster_alms=config.pre_proc_pars.DEBUGHARMONICuse_namaster_alms,
    )
    logger.info(f"Pre-processed alms have shape: {freq_alms_convolved.shape}")

    if config.pre_proc_pars.correct_for_TF:
        inv_sqrt_tf_lm_freq = np.load(
            manager.get_path_to_preprocessed_reduced_TF(), allow_pickle=True
        )[config.pre_proc_pars.DEBUGHARMONIC_preproc_invTFLM_to_use]
        # nmt_bins = load_nmt_binning(manager)

        # inv_sqrt_tf_bin_freq, inv_sqrt_tf_freq, inv_sqrt_tf_lm_freq = (
        #     get_tf_alm_correction_fora_all_channels(
        #         manager, config, nmt_bins, use_new_reduced_TF=True
        #     )
        # )
        freq_alms_convolved = np.einsum("fijl,fjl->fil", inv_sqrt_tf_lm_freq, freq_alms_convolved)

        # print("TESTTTTTTINNNNNNNNNNGGGGGGG")
        # inv_sqrt_tf_bin_freq, inv_sqrt_tf_freq, inv_sqrt_tf_lm_freq = get_effectiv_inv_tf_least_square(manager, config, nmt_bins)
        # inv_sqrt_tf_lm_freq = np.load('/shared_home/jost/code/notebooks/test_sims_filtering/save_inv_T_ellem_freq.npy')
        # freq_alms_convolved = np.einsum("flij,fjl->fil", inv_sqrt_tf_lm_freq, freq_alms_convolved)

        # Saving the reduced transfer functions for later use
        # TODO: save TF in a seperate function like the other outputs
        # fname_TF = manager.get_path_to_preprocessed_reduced_TF()
        # fname_TF.parent.mkdir(parents=True, exist_ok=True)

        # np.savez(
        #     fname_TF,
        #     inv_sqrt_tf_bin_freq=inv_sqrt_tf_bin_freq,
        #     inv_sqrt_tf_freq=inv_sqrt_tf_freq,
        #     inv_sqrt_tf_lm_freq=inv_sqrt_tf_lm_freq,
        # )
        # logger.info(
        #     f"Saved reduced transfer functions to {manager.get_path_to_preprocessed_reduced_TF()}"
        # )

    if mask_output and not config.parametric_sep_pars.use_harmonic_compsep:
        binary_mask = hp.read_map(manager.path_to_binary_mask)
        freq_maps_convolved = apply_binary_mask(freq_maps_convolved, binary_mask=binary_mask)
    if config.parametric_sep_pars.use_harmonic_compsep:
        return freq_maps_convolved, freq_alms_convolved
    return freq_maps_convolved


def save_preprocessed_maps(manager: DataManager, freq_maps, id_sim: int | None = None):
    fname = manager.get_path_to_preprocessed_maps(id_sim)
    logger.info(f"Saving pre-processed maps to {fname}")
    np.save(fname, freq_maps)


def save_preprocessed_alms(manager: DataManager, freq_alms, id_sim: int | None = None):
    fname = manager.get_path_to_preprocessed_alms(id_sim)
    logger.info(f"Saving pre-processed alms to {fname}")
    np.save(fname, freq_alms)


def preproc_and_save(config: Config, manager: DataManager, id_sim: int | None = None) -> None | int:
    with Timer("preprocesser"):
        convolved_output = preprocess_map(manager, config, id_sim=id_sim)
    if config.parametric_sep_pars.use_harmonic_compsep:
        save_preprocessed_maps(manager, convolved_output[0], id_sim=id_sim)
        save_preprocessed_alms(manager, convolved_output[1], id_sim=id_sim)
    else:
        save_preprocessed_maps(manager, convolved_output, id_sim=id_sim)
    return id_sim


def main():
    parser = argparse.ArgumentParser(description="Preprocesser")
    parser.add_argument("--config", type=Path, required=True, help="config file")
    parser.add_argument("--sim", type=int, default=None, help="process only this simulation index")

    args = parser.parse_args()
    config = Config.load_yaml(args.config)
    manager = DataManager(config)

    world, rank, size = get_world()
    if rank == 0:
        manager.dump_config()
        manager.create_output_dirs(config.map_sim_pars.n_sim, config.noise_sim_pars.n_sim)

    if args.sim is not None:
        preproc_and_save(config, manager, id_sim=args.sim)
        return

    n_sim_sky = config.map_sim_pars.n_sim
    if n_sim_sky == 0:  # No sky simulations: run preprocessing on the real data
        preproc_and_save(config, manager, id_sim=None)
    elif size < 2:
        for i in range(n_sim_sky):
            result = preproc_and_save(config, manager, id_sim=i)
            logger.info(f"Finished preprocessing map {result + 1} / {n_sim_sky}")
    else:
        from mpi4py.futures import MPICommExecutor

        with MPICommExecutor() as executor:
            if executor is not None:
                logger.info(f"Distributing work to {executor.num_workers} workers")
                func = partial(preproc_and_save, config, manager)
                for result in executor.map(func, range(n_sim_sky), unordered=True):
                    logger.info(f"Finished preprocessing map {result + 1} / {n_sim_sky}")


if __name__ == "__main__":
    main()
