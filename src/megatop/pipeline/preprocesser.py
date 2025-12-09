import argparse
from functools import partial
from pathlib import Path

import healpy as hp
import numpy as np
from mpi4py.futures import MPICommExecutor
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


def get_reduced_tf(transfer, nmt_bins_native, sum_TF_column=False):
    inv_sqrt_tf_full = np.linalg.inv([sqrtm(TF_ell.T) for TF_ell in transfer.T])[
        :, -4:, -4:
    ]  # keeping only polarised components

    inv_sqrt_tf_bin = np.zeros((2, 2, inv_sqrt_tf_full.shape[0]), dtype=np.complex128)
    if not sum_TF_column:  # Rearanging diagonal elements only
        inv_sqrt_tf_bin[0, 0] = inv_sqrt_tf_full[:, 0, 0]
        inv_sqrt_tf_bin[0, 1] = inv_sqrt_tf_full[:, 1, 1]
        inv_sqrt_tf_bin[1, 0] = inv_sqrt_tf_full[:, 2, 2]
        inv_sqrt_tf_bin[1, 1] = inv_sqrt_tf_full[:, 3, 3]
    else:
        inv_sqrt_tf_sum = np.sum(inv_sqrt_tf_full, axis=1)
        inv_sqrt_tf_bin[0, 0] = inv_sqrt_tf_sum[:, 0]
        inv_sqrt_tf_bin[0, 1] = inv_sqrt_tf_sum[:, 1]
        inv_sqrt_tf_bin[1, 0] = inv_sqrt_tf_sum[:, 2]
        inv_sqrt_tf_bin[1, 1] = inv_sqrt_tf_sum[:, 3]

    inv_sqrt_tf = np.zeros((2, 2, nmt_bins_native.lmax + 1), dtype=np.complex128)
    for i in range(2):
        for j in range(2):
            inv_sqrt_tf[i, j] = homemade_unbin_cell(inv_sqrt_tf_bin[i, j], nmt_bins_native)

    return inv_sqrt_tf, inv_sqrt_tf_bin


def get_reduced_tf_new(transfer, nmt_bins_native):
    inv_transfer = np.linalg.inv(transfer[-4:, -4:].T).T

    corner_TF = np.zeros((2, 2, inv_transfer.shape[2]))
    corner_TF[0, 0, :] = inv_transfer[-4, -4, :]
    corner_TF[1, 1, :] = inv_transfer[-1, -1, :]
    corner_TF[0, 1, :] = inv_transfer[-4, -1, :]
    corner_TF[1, 0, :] = inv_transfer[-1, -4, :]

    reduced_inv_tf_binned = np.sqrt(corner_TF + 0j)  # element wise sqrt

    reduced_inv_tf = np.zeros((2, 2, nmt_bins_native.lmax + 1), dtype=np.complex128)
    for i in range(2):
        for j in range(2):
            reduced_inv_tf[i, j] = homemade_unbin_cell(reduced_inv_tf_binned[i, j], nmt_bins_native)

    return reduced_inv_tf, reduced_inv_tf_binned


def preprocess_map(
    manager: DataManager, config: Config, id_sim: int | None = None, mask_output=True
):
    input_maps = read_input_maps(manager.get_maps_filenames(sub=id_sim))
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
        )
        logger.info(f"Pre-processed maps have shape: {freq_maps_convolved.shape}")

    if config.parametric_sep_pars.use_harmonic_compsep and not skip_preprocessing_bool:
        logger.info(
            "Using harmonic pipeline for component separation. Pre-processing will output alms"
        )
        analysis_mask = hp.read_map(manager.path_to_analysis_mask)

        logger.warning("Normalizing analysis mask to 1, TODO: remove after merge")
        # TODO: remove after merge
        analysis_mask /= np.max(analysis_mask)  # normalize the mask to 1
        binary_mask = hp.read_map(manager.path_to_binary_mask)

        if config.pre_proc_pars.DEBUGHARMONICuse_namaster_alms:
            mask_alm_computation = analysis_mask
        else:
            mask_alm_computation = binary_mask

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
            analysis_mask=mask_alm_computation,
            harmonic_analysis_lmax=config.parametric_sep_pars.harmonic_lmax,
            purify_e=config.map2cl_pars.purify_e,
            purify_b=config.map2cl_pars.purify_b,
            use_namaster_alms=config.pre_proc_pars.DEBUGHARMONICuse_namaster_alms,
        )
        logger.info(f"Pre-processed alms have shape: {freq_alms_convolved.shape}")

        if config.pre_proc_pars.correct_for_TF:
            nmt_bins_native = load_nmt_binning(manager)
            lm_size = freq_alms_convolved.shape[-1]

            logger.warning("Including transfer function in the pre-processed alms. ")
            inv_sqrt_tf_bin_freq = np.zeros(
                (len(manager.get_TF_filenames()), 2, 2, nmt_bins_native.get_n_bands()),
                dtype=np.complex128,
            )
            inv_sqrt_tf_freq = np.zeros(
                (len(manager.get_TF_filenames()), 2, 2, config.parametric_sep_pars.harmonic_lmax),
                dtype=np.complex128,
            )
            inv_sqrt_tf_lm_freq = np.zeros(
                (len(manager.get_TF_filenames()), 2, 2, lm_size), dtype=np.complex128
            )

            for f, tf_path in enumerate(manager.get_TF_filenames()):
                if tf_path == Path():
                    logger.warning(
                        f"DEBUG: Transfer function for frequency {config.frequencies[f]} is not provided, skipping."
                    )
                    continue
                logger.info(f"Loading transfer function from {tf_path}")
                # Loading TF:
                transfer = np.load(tf_path, allow_pickle=True)["full_tf"]

                """
                inv_sqrt_tf_full = np.linalg.inv([sqrtm(TF_ell.T) for TF_ell in transfer.T])[
                    :, -4:, -4:
                ]  # keeping only polarised compoenents

                # Keeping onky the following elements:
                #  EE->EE  ;  EB->EB
                #  BE->BE  ;  BB->BB
                #
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
                """
                # import IPython; IPython.embed()
                # inv_sqrt_tf, inv_sqrt_tf_bin = get_reduced_tf(
                #     transfer, nmt_bins_native, config.pre_proc_pars.sum_TF_column
                # )
                logger.warning("TESTING NEW METHOD FOR REDUCED TF REDUCTION")
                inv_sqrt_tf, inv_sqrt_tf_bin = get_reduced_tf_new(transfer, nmt_bins_native)
                inv_sqrt_tf = inv_sqrt_tf[..., : config.parametric_sep_pars.harmonic_lmax]

                inv_sqrt_tf_bin_freq[f] = inv_sqrt_tf_bin
                inv_sqrt_tf_freq[f] = inv_sqrt_tf

                inv_sqrt_tf_lm = np.zeros((2, 2, lm_size), dtype=np.complex128)
                for index in range(lm_size):
                    inv_sqrt_tf_lm[..., index] = inv_sqrt_tf[
                        ..., hp.Alm.getlm(lmax=hp.Alm.getlmax(lm_size), i=index)[0]
                    ]

                inv_sqrt_tf_lm_freq[f] = inv_sqrt_tf_lm

                # Applying the transfer function to the alms
                freq_alms_convolved[f] = np.einsum(
                    "ijl,jl->il", inv_sqrt_tf_lm, freq_alms_convolved[f]
                )

            # Saving the reduced transfer functions for later use
            # TODO: save TF in a seperate function like the other outputs
            fname_TF = manager.get_path_to_preprocessed_reduced_TF()
            fname_TF.parent.mkdir(parents=True, exist_ok=True)

            np.savez(
                fname_TF,
                inv_sqrt_tf_bin_freq=inv_sqrt_tf_bin_freq,
                inv_sqrt_tf_freq=inv_sqrt_tf_freq,
                inv_sqrt_tf_lm_freq=inv_sqrt_tf_lm_freq,
            )
            logger.info(
                f"Saved reduced transfer functions to {manager.get_path_to_preprocessed_reduced_TF()}"
            )
            # import IPython; IPython.embed()
    if mask_output and not config.parametric_sep_pars.use_harmonic_compsep:
        binary_mask = hp.read_map(manager.path_to_binary_mask)
        freq_maps_convolved = apply_binary_mask(freq_maps_convolved, binary_mask=binary_mask)
    if config.parametric_sep_pars.use_harmonic_compsep:
        return freq_maps_convolved, freq_alms_convolved
    return freq_maps_convolved


def save_preprocessed_maps(manager: DataManager, freq_maps, id_sim: int | None = None):
    fname = manager.get_path_to_preprocessed_maps(sub=id_sim)
    fname.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving pre-processed maps to {fname}")
    np.save(fname, freq_maps)


def save_preprocessed_alms(manager: DataManager, freq_alms, id_sim: int | None = None):
    fname = manager.get_path_to_preprocessed_alms(sub=id_sim)
    fname.parent.mkdir(parents=True, exist_ok=True)
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
    parser = argparse.ArgumentParser(
        description="Preprocesser", epilog="mpi4py is required to run this script"
    )
    parser.add_argument("--config", type=Path, required=True, help="config file")

    args = parser.parse_args()
    config = Config.load_yaml(args.config)
    manager = DataManager(config)

    world, rank, size = get_world()
    if rank == 0:
        manager.dump_config()

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
