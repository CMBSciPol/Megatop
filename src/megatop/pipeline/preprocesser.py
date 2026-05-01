import argparse
from functools import partial
from pathlib import Path

import healpy as hp
import numpy as np
from scipy.linalg import sqrtm

from megatop import Config, DataManager
from megatop.utils import Timer, logger
from megatop.utils.binning import load_nmt_binning
from megatop.utils.mask import apply_binary_mask
from megatop.utils.mpi import get_world
from megatop.utils.preproc import alm_common_beam, common_beam_and_nside, read_input_maps


def homemade_unbin_cell(binned_cell, nmt_bins):
    """
    Necessary to unbin cell that are complex numbers
    """
    output_shape = np.array(binned_cell.shape)
    output_shape[-1] = nmt_bins.lmax + 1
    unbinned_cell = np.zeros(output_shape, dtype=np.complex128)
    for bin in range(binned_cell.shape[-1]):
        unbinned_cell[..., nmt_bins.get_ell_list(bin)] = binned_cell[..., bin]
    return unbinned_cell


def preprocess_map(
    manager: DataManager, config: Config, id_sim: int | None = None, mask_output=True
):
    input_maps = read_input_maps(manager.get_maps_filenames(id_sim))
    logger.info(
        f"Input maps have shapes: {[input_maps[i].shape for i in range(len(config.frequencies))]}"
    )

    # bool for the different conditions where the preprocessing can be skipped e.g. debug, no beam, same input and common beam, etc.
    skip_preprocessing_bool = (
        np.all(np.array(config.pre_proc_pars.common_beam_correction) == np.array(config.beams))
        or config.pre_proc_pars.DEBUGskippreproc
    ) and not config.parametric_sep_pars.use_harmonic_compsep

    if skip_preprocessing_bool:  # and not DEBUGtruncatealms:
        logger.info("Common beam correction is the same as the input beam, no need to apply it.")
        logger.warning("This is mostly for testing it might not actually represent the real noise")
        freq_maps_convolved = np.array(input_maps, dtype="float64")

    else:
        freq_maps_convolved = common_beam_and_nside(
            nside=config.nside,
            common_beam=config.pre_proc_pars.common_beam_correction,
            frequency_beams=config.beams,
            freq_maps=input_maps,
            lmax=config.lmax,
        )
        logger.info(f"Pre-processed maps have shape: {freq_maps_convolved.shape}")

    if config.parametric_sep_pars.use_harmonic_compsep and not skip_preprocessing_bool:
        logger.info(
            "Using harmonic pipeline for component separation. Pre-processing will output alms"
        )
        analysis_mask = hp.read_map(manager.path_to_analysis_mask)

        freq_beams = config.beams
        common_beam = config.pre_proc_pars.common_beam_correction
        if config.pre_proc_pars.DEBUGskippreproc:
            freq_beams = np.array([0.0] * len(config.frequencies))
            common_beam = 0.0
        freq_alms_convolved = alm_common_beam(
            nside=config.nside,
            common_beam=common_beam,
            frequency_beams=freq_beams,
            freq_maps=np.array(input_maps),
            analysis_mask=analysis_mask,
            harmonic_analysis_lmax=config.parametric_sep_pars.harmonic_lmax,
        )
        logger.info(f"Pre-processed alms have shape: {freq_alms_convolved.shape}")

        if config.pre_proc_pars.correct_for_TF:
            logger.warning("Including transfer function in the pre-processed alms. ")
            for f, tf_path in enumerate(manager.get_TF_filenames()):
                if tf_path is None:
                    logger.warning(
                        f"DEBUG: Transfer function for frequency {config.frequencies[f]} is not provided, skipping."
                    )
                    continue
                logger.info(f"Loading transfer function from {tf_path}")
                # Loading TF:
                transfer = np.load(tf_path, allow_pickle=True)["full_tf"]

                nmt_bins_native = load_nmt_binning(manager)
                inv_sqrt_tf_full = np.linalg.inv([sqrtm(TF_ell.T) for TF_ell in transfer.T])[
                    :, -4:, -4:
                ]  # keeping only polarised compoenents

                # Keeping onky the following elements:
                #  EE->EE  ;  EB->EB
                #  BE->BE  ;  BB->BB
                #
                # import IPython; IPython.embed()
                """
                inv_sqrt_tf = np.zeros((2, 2, nmt_bins_native.lmax + 1), dtype=np.complex128)
                inv_sqrt_tf[0, 0] = homemade_unbin_cell(
                    inv_sqrt_tf_full[:, 0, 0], nmt_bins_native
                )  # EE->EE
                inv_sqrt_tf[0, 1] = homemade_unbin_cell(
                    inv_sqrt_tf_full[:, 1, 1], nmt_bins_native
                )  # EB->EB
                # inv_sqrt_tf[0, 1] = homemade_unbin_cell(
                #     inv_sqrt_tf_full[:, 0, -1], nmt_bins_native
                # )  # EB->EB
                inv_sqrt_tf[1, 0] = homemade_unbin_cell(
                    inv_sqrt_tf_full[:, 2, 2], nmt_bins_native
                )  # BE->BE
                # inv_sqrt_tf[1, 0] = homemade_unbin_cell(
                #     inv_sqrt_tf_full[:, -1, 0], nmt_bins_native
                # )  # BE->BE
                inv_sqrt_tf[1, 1] = homemade_unbin_cell(
                    inv_sqrt_tf_full[:, 3, 3], nmt_bins_native
                )  # BB->BB
                inv_sqrt_tf = inv_sqrt_tf[..., : config.parametric_sep_pars.harmonic_lmax]
                """
                inv_sqrt_tf_bin = np.zeros((2, 2, inv_sqrt_tf_full.shape[0]), dtype=np.complex128)
                inv_sqrt_tf_bin[0, 0] = inv_sqrt_tf_full[:, 0, 0]
                inv_sqrt_tf_bin[0, 1] = inv_sqrt_tf_full[:, 1, 1]
                inv_sqrt_tf_bin[1, 0] = inv_sqrt_tf_full[:, 2, 2]
                inv_sqrt_tf_bin[1, 1] = inv_sqrt_tf_full[:, 3, 3]

                inv_sqrt_tf = np.zeros((2, 2, nmt_bins_native.lmax + 1), dtype=np.complex128)
                for i in range(2):
                    for j in range(2):
                        inv_sqrt_tf[i, j] = homemade_unbin_cell(
                            inv_sqrt_tf_bin[i, j], nmt_bins_native
                        )
                inv_sqrt_tf = inv_sqrt_tf[..., : config.parametric_sep_pars.harmonic_lmax]

                if config.pre_proc_pars.sum_TF_column:
                    logger.warning(
                        "DEBUG: Summing over columns of the transfer instead of rearanging diagonal elements"
                    )
                    inv_sqrt_tf_sum = np.sum(
                        inv_sqrt_tf_full, axis=1
                    )  # summing over column to get all the contribution xy-->ab (EE-->EE + EB-->EE + BE-->EE + BB-->EE etc)
                    inv_sqrt_tf = np.zeros((2, 2, nmt_bins_native.lmax + 1), dtype=np.complex128)
                    inv_sqrt_tf[0, 0] = homemade_unbin_cell(
                        inv_sqrt_tf_sum[:, 0], nmt_bins_native
                    )  # EE->EE
                    inv_sqrt_tf[0, 1] = homemade_unbin_cell(
                        inv_sqrt_tf_sum[:, 1], nmt_bins_native
                    )  # EB->EB
                    inv_sqrt_tf[1, 0] = homemade_unbin_cell(
                        inv_sqrt_tf_sum[:, 2], nmt_bins_native
                    )  # BE->BE
                    inv_sqrt_tf[1, 1] = homemade_unbin_cell(
                        inv_sqrt_tf_sum[:, 3], nmt_bins_native
                    )  # BB->BB

                lm_size = freq_alms_convolved.shape[-1]
                inv_sqrt_tf_lm = np.zeros((2, 2, lm_size), dtype=np.complex128)
                for index in range(lm_size):
                    inv_sqrt_tf_lm[..., index] = inv_sqrt_tf[
                        ..., hp.Alm.getlm(lmax=hp.Alm.getlmax(lm_size), i=index)[0]
                    ]

                # Applying the transfer function to the alms
                freq_alms_convolved[f] = np.einsum(
                    "ijl,jl->il", inv_sqrt_tf_lm, freq_alms_convolved[f]
                )

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
