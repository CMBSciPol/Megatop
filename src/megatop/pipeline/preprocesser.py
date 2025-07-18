import argparse
from functools import partial
from pathlib import Path

import healpy as hp
import numpy as np
import pymaster as nmt
from mpi4py.futures import MPICommExecutor
from scipy.linalg import sqrtm

from megatop import Config, DataManager
from megatop.utils import Timer, logger
from megatop.utils.mask import apply_binary_mask
from megatop.utils.mpi import get_world
from megatop.utils.preproc import alm_common_beam, common_beam_and_nside, read_input_maps


def homemade_unbin_cell(binned_cell, nmt_bins):
    output_shape = np.array(binned_cell.shape)
    output_shape[-1] = nmt_bins.lmax + 1
    unbinned_cell = np.zeros(output_shape, dtype=np.complex128)
    for bin in range(binned_cell.shape[-1]):
        unbinned_cell[..., nmt_bins.get_ell_list(bin)] = binned_cell[..., bin]
    return unbinned_cell


def preprocess_map(
    manager: DataManager, config: Config, id_sim: int | None = None, mask_output=True
):
    input_maps = read_input_maps(manager.get_maps_filenames(sub=id_sim))
    logger.info(
        f"Input maps have shapes: {[input_maps[i].shape for i in range(len(config.frequencies))]}"
    )

    # DEBUGtruncatealms = True,
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
        # DEBUGlm_range= [config.parametric_sep_pars.harmonic_lmin, config.parametric_sep_pars.harmonic_lmax]
        freq_maps_convolved = common_beam_and_nside(
            nside=config.nside,
            common_beam=config.pre_proc_pars.common_beam_correction,
            frequency_beams=config.beams,
            freq_maps=input_maps,
            # DEBUGtruncatealms=DEBUGtruncatealms,
            # DEBUGlm_range=DEBUGlm_range,
        )
        logger.info(f"Pre-processed maps have shape: {freq_maps_convolved.shape}")

    if (
        config.parametric_sep_pars.use_harmonic_compsep and not skip_preprocessing_bool
    ):  # and config.parametric_sep_pars.DEBUGnamaster_deconv:# and not config.parametric_sep_pars.DEBUGcommon_beam_correction_before_smoothmask:
        logger.info(
            "Using harmonic pipeline for component separation. Pre-processing will output alms"
        )
        analysis_mask = hp.read_map(manager.path_to_analysis_mask)

        if config.masks_pars.DEBUG_output_apod_binary_mask:
            logger.warning("DEBUG: Using apodized binary mask for harmonic component separation, ")
            analysis_mask = hp.read_map(manager.path_to_apod_binary_mask)

        logger.warning("Normalizing analysis mask to 1, TODO: remove after merge")
        # TODO: remove after merge
        analysis_mask /= np.max(analysis_mask)  # normalize the mask to 1

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

        if config.pre_proc_pars.DEBUGinclude_TF:
            # import IPython; IPython.embed()
            logger.warning("DEBUG: Including transfer function in the pre-processed alms. ")
            for f, tf_path in enumerate(manager.get_TF_filenames()):
                if tf_path == Path():
                    logger.warning(
                        f"DEBUG: Transfer function for frequency {config.frequencies[f]} is not provided, skipping."
                    )
                    continue
                logger.info(f"Loading transfer function from {tf_path}")
                transfer = np.load(tf_path, allow_pickle=True)["tf"]
                # Loading TF:
                # BBTF = np.load(
                #     "/lustre/work/jost/SO_MEGATOP/harmonic_test_Nl_std_beam_nhits_obsmatBBMASER_namaster/TF_FirstDayEveryMonth_Full_nside512_fpthin8_pwf_beam.npz",
                #     allow_pickle=True,
                # )
                # transfer = BBTF["tf"]
                nside_native = 512
                # import IPython; IPython.embed()
                nmt_bins_native = nmt.NmtBin.from_nside_linear(nside_native, nlb=10, is_Dell=False)
                inv_sqrt_tf_full = np.linalg.inv([sqrtm(TF_ell.T) for TF_ell in transfer.T])
                # Keeping onky the following elements:
                #  EE->EE  ;  EB->EB
                #  BE->BE  ;  BB->BB
                #
                # import IPython; IPython.embed()
                inv_sqrt_tf = np.zeros((2, 2, nmt_bins_native.lmax + 1), dtype=np.complex128)
                inv_sqrt_tf[0, 0] = homemade_unbin_cell(
                    inv_sqrt_tf_full[:, 0, 0], nmt_bins_native
                )  # EE->EE
                inv_sqrt_tf[0, 1] = homemade_unbin_cell(
                    inv_sqrt_tf_full[:, 1, 1], nmt_bins_native
                )  # EB->EB
                inv_sqrt_tf[1, 0] = homemade_unbin_cell(
                    inv_sqrt_tf_full[:, 2, 2], nmt_bins_native
                )  # BE->BE
                inv_sqrt_tf[1, 1] = homemade_unbin_cell(
                    inv_sqrt_tf_full[:, 3, 3], nmt_bins_native
                )  # BB->BB
                inv_sqrt_tf = inv_sqrt_tf[..., : config.parametric_sep_pars.harmonic_lmax]

                DEBUG_summ_column = True
                if DEBUG_summ_column:
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

                # DEBUGTEST93145_TF_ONLY = False
                # if DEBUGTEST93145_TF_ONLY:
                #     logger.warning(
                #         "DEBUG: ONLY USING TF ON 93 and 145 GHz, !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                #         "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                #         )
                #     freq_inv_sqrt_tf_lm = np.zeros((len(config.frequencies), 2, 2, lm_size), dtype=np.complex128)
                #     freq_inv_sqrt_tf_lm[0, 0,0,:] = 1
                #     freq_inv_sqrt_tf_lm[0, 1,1,:] = 1
                #     freq_inv_sqrt_tf_lm[1, 0,0,:] = 1
                #     freq_inv_sqrt_tf_lm[1, 1,1,:] = 1
                #     freq_inv_sqrt_tf_lm[2] = inv_sqrt_tf_lm
                #     freq_inv_sqrt_tf_lm[3] = inv_sqrt_tf_lm
                #     freq_inv_sqrt_tf_lm[4, 0,0,:] = 1
                #     freq_inv_sqrt_tf_lm[4, 1,1,:] = 1
                #     freq_inv_sqrt_tf_lm[5, 0,0,:] = 1
                #     freq_inv_sqrt_tf_lm[5, 1,1,:] = 1
                # else:
                #     freq_inv_sqrt_tf_lm = np.zeros((len(config.frequencies), 2, 2, lm_size), dtype=np.complex128)
                #     for f in range(len(config.frequencies)):
                #         freq_inv_sqrt_tf_lm[f] = inv_sqrt_tf_lm

                # Applying the transfer function to the alms
                # freq_alms_convolved_TF_corrected = np.zeros_like(freq_alms_convolved, dtype=np.complex128)
                # for f in range(len(config.frequencies)):
                #     freq_alms_convolved_TF_corrected[f] = np.einsum('ijl,jl->il', freq_inv_sqrt_tf_lm[f], freq_alms_convolved[f])
                freq_alms_convolved[f] = np.einsum(
                    "ijl,jl->il", inv_sqrt_tf_lm, freq_alms_convolved[f]
                )
                # freq_alms_convolved = freq_alms_convolved_TF_corrected
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
