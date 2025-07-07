import argparse
from functools import partial
from pathlib import Path

import healpy as hp
import numpy as np
from mpi4py.futures import MPICommExecutor

from megatop import Config, DataManager
from megatop.utils import Timer, logger
from megatop.utils.mask import apply_binary_mask
from megatop.utils.preproc import alm_common_beam, common_beam_and_nside, read_input_maps


def preprocess_map(
    manager: DataManager, config: Config, id_sim: int | None = None, mask_output=True
):
    input_maps = read_input_maps(manager.get_maps_filenames(sub=id_sim))
    logger.info(
        f"Input maps have shapes: {[input_maps[i].shape for i in range(len(config.frequencies))]}"
    )

    # DEBUGtruncatealms = True,

    if (
        np.all(np.array(config.pre_proc_pars.common_beam_correction) == np.array(config.beams))
        or config.pre_proc_pars.DEBUGskippreproc
    ) and not config.parametric_sep_pars.use_harmonic_compsep:  # and not DEBUGtruncatealms:
        logger.info("Common beam correction is the same as the input beam, no need to apply it.")
        logger.warning("This is mostly for testing it might not actually represent the real noise")
        freq_maps_convolved = np.array(input_maps, dtype="float64")
    elif config.parametric_sep_pars.use_harmonic_compsep:  # and config.parametric_sep_pars.DEBUGnamaster_deconv:# and not config.parametric_sep_pars.DEBUGcommon_beam_correction_before_smoothmask:
        logger.info(
            "Using harmonic pipeline for component separation. Pre-processing will output alms"
        )
        analysis_mask = hp.read_map(manager.path_to_analysis_mask)
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

    if mask_output and not config.parametric_sep_pars.use_harmonic_compsep:
        binary_mask = hp.read_map(manager.path_to_binary_mask)
        freq_maps_convolved = apply_binary_mask(freq_maps_convolved, binary_mask=binary_mask)
    if config.parametric_sep_pars.use_harmonic_compsep:
        return freq_alms_convolved
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
        save_preprocessed_alms(manager, convolved_output, id_sim=id_sim)
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
