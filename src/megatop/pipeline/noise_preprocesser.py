"""Per-realisation noise preprocessing for the pixel noisecov pipeline.

Splits the per-realisation work out of `pixel_noisecov_estimater` so Snakemake
can fan out one job per noise realisation. Each invocation:

* reads one noise realisation (all frequencies),
* applies the common-beam / nside preprocessing,
* saves the preprocessed noise maps to disk,
* (if `use_harmonic_compsep`) computes the namaster noise spectra and the
  TF-corrected ``nl`` contribution, then saves the binned + unbinned spectra.

The aggregator (``megatop-noisecov-run``) then averages these per-realisation
contributions into the final pixel + harmonic noise covariance.
"""

import argparse
import os
from pathlib import Path

import healpy as hp
import numpy as np

from megatop import Config, DataManager

# from megatop.pipeline.preprocesser import get_tf_alm_correction_fora_all_channels
from megatop.pipeline.map_to_cler import get_inv_binned_MCM
from megatop.utils import Timer, logger
from megatop.utils.binning import load_nmt_binning
from megatop.utils.mpi import get_world
from megatop.utils.preproc import alm_common_beam, common_beam_and_nside, read_input_maps
from megatop.utils.spectra import alm_2_cl_MCM, initialize_nmt_workspace, spectra_from_namaster

# from megatop.utils.TF_utils import get_reduced_TF, rebuilt_from_reduced_conj

HEALPY_DATA_PATH = os.getenv("HEALPY_LOCAL_DATA", None)


def _preprocess_noise_maps(
    config: Config, manager: DataManager, id_real: int | None, TRUE_noise_maps: bool = False
) -> np.ndarray:
    noise_freq_maps = []
    noise_maps_filenames_list = (
        manager.get_TRUE_noise_maps_filenames(id_real)
        if TRUE_noise_maps
        else manager.get_noise_maps_filenames(id_real)
    )
    for noise_filename in noise_maps_filenames_list:
        msg = "Importing TRUE noise map" if TRUE_noise_maps else "Importing noise map"
        logger.debug(f"{msg}: {noise_filename}")
        noise_freq_maps.append(hp.read_map(noise_filename, field=None, dtype=np.float64))

    # Always go through common_beam_and_nside even when common_beam == beams (no actual beam
    # correction). The map2alm→alm2map cycle bandlimits pixel-space noise maps to config.lmax,
    # preventing aliasing from modes above lmax into the analysis bins.
    return common_beam_and_nside(
        nside=config.nside,
        common_beam=config.pre_proc_pars.common_beam_correction,
        frequency_beams=config.beams,
        freq_maps=noise_freq_maps,
        lmax=config.lmax,
    )


def _harmonic_alm_preproc_contrib(
    config: Config, manager: DataManager, id_real: int | None
) -> np.ndarray:
    input_noise_maps = read_input_maps(manager.get_noise_maps_filenames(id_real))
    logger.info(
        f"Input noise maps have shapes: {[input_noise_maps[i].shape for i in range(len(config.frequencies))]}"
    )
    analysis_mask = hp.read_map(manager.path_to_analysis_mask)

    freq_alms_convolved = alm_common_beam(
        common_beam=config.pre_proc_pars.common_beam_correction,
        frequency_beams=config.beams,
        freq_maps=np.array(input_noise_maps),
        lmax=config.lmax,
        analysis_mask=analysis_mask,
        purify_e=config.map2cl_pars.purify_e,
        purify_b=config.map2cl_pars.purify_b,
        use_namaster_alms=config.pre_proc_pars.DEBUGHARMONICuse_namaster_alms,
    )

    if config.pre_proc_pars.correct_for_TF:
        # nmt_bins = load_nmt_binning(manager)
        inv_sqrt_tf_lm_freq = np.load(
            manager.get_path_to_preprocessed_reduced_TF(), allow_pickle=True
        )[config.pre_proc_pars.DEBUGHARMONIC_preproc_invTFLM_to_use]
        # inv_sqrt_tf_bin_freq, inv_sqrt_tf_freq, inv_sqrt_tf_lm_freq = (
        #     get_tf_alm_correction_fora_all_channels(
        #         manager, config, nmt_bins, use_new_reduced_TF=True
        #     )
        # )
        freq_alms_convolved = np.einsum("fijl,fjl->fil", inv_sqrt_tf_lm_freq, freq_alms_convolved)

        # print("TESTTTTTTINNNNNNNNNNGGGGGGG")
        # inv_sqrt_tf_lm_freq = np.load('/shared_home/jost/code/notebooks/test_sims_filtering/save_inv_T_ellem_freq.npy')
        # freq_alms_convolved = np.einsum("flij,fjl->fil", inv_sqrt_tf_lm_freq, freq_alms_convolved)

    return freq_alms_convolved


def _harmonic_nl_from_alms_contrib(
    config: Config,
    manager: DataManager,
    freq_alms_convolved: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    analysis_mask = hp.read_map(manager.path_to_analysis_mask, dtype=np.float64)

    nmt_bins = load_nmt_binning(manager)
    workspace_nmt = initialize_nmt_workspace(
        nmt_bins=nmt_bins,
        analysis_mask=analysis_mask,
        beam=None,
        purify_e=config.map2cl_pars.purify_e,
        purify_b=config.map2cl_pars.purify_b,
        n_iter=config.map2cl_pars.n_iter_namaster,
        lmax=config.lmax,
    )
    inv_MCM = get_inv_binned_MCM(workspace_nmt, nmt_bins)
    nl_from_alm_decoupled_freq = []
    nl_from_alm_decoupled_freq_unbinned = []
    for f in range(freq_alms_convolved.shape[0]):
        nl_from_alm_decoupled_freq.append(
            alm_2_cl_MCM(
                freq_alms_convolved[f],
                freq_alms_convolved[f],
                inv_MCM,
                config.lmax,
                nmt_bins,
                effective_beam_CMB=None,
            )
        )
        nl_from_alm_decoupled_freq_unbinned.append(
            nmt_bins.unbin_cell(nl_from_alm_decoupled_freq[-1])
        )
    nl_from_alm_decoupled_freq = np.array(nl_from_alm_decoupled_freq)
    nl_from_alm_decoupled_freq_unbinned = np.array(nl_from_alm_decoupled_freq_unbinned)
    # nl_from_alm_decoupled_freq_unbinned = nmt_bins.unbin_cell(nl_from_alm_decoupled_freq)

    return nl_from_alm_decoupled_freq, nl_from_alm_decoupled_freq_unbinned


def _harmonic_nl_contrib(
    config: Config,
    manager: DataManager,
    noise_freq_maps_preprocessed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the harmonic ``nl`` contribution (binned + unbinned) for one realisation."""
    nmt_bins = load_nmt_binning(manager)
    # bin_index_lminlmax = np.load(manager.path_to_binning, allow_pickle=True)["bin_index_lminlmax"]

    mask_analysis = hp.read_map(manager.path_to_analysis_mask, dtype=np.float64)

    with Timer("init-namaster-workspace"):
        workspaceff = initialize_nmt_workspace(
            nmt_bins=nmt_bins,
            analysis_mask=mask_analysis,
            beam=None,
            purify_e=config.map2cl_pars.purify_e,
            purify_b=config.map2cl_pars.purify_b,
            n_iter=10,
            lmax=config.lmax,
        )

    noise_spectra_with_cross, noise_spectra_unbined_with_cross = spectra_from_namaster(
        noise_freq_maps_preprocessed,
        mask_analysis,
        workspaceff,
        nmt_bins,
        compute_cross_freq=False,
        purify_e=config.map2cl_pars.purify_e,
        purify_b=config.map2cl_pars.purify_b,
        beam=None,
        return_all_spectra=True,  # config.pre_proc_pars.correct_for_TF,
        lmax=config.lmax,
    )

    if config.pre_proc_pars.correct_for_TF:
        logger.warning("Including transfer function in the pre-processed noise spectra.")

        # output_noise_spectra_with_cross = np.zeros(
        #     [len(config.frequencies), 4, nmt_bins.get_n_bands()]
        # )
        # output_noise_spectra_unbined_with_cross = np.zeros(
        #     [len(config.frequencies), 4, noise_spectra_unbined_with_cross.shape[-1]]
        # )
        # use_new_reduced_TF = True
        # use_full_inv = False
        logger.warning("TEST: REMOVE THIS FLAG WHEN THE NEW METHOD IS STABLE")
        # for f, tf_path in enumerate(manager.get_TF_filenames()):
        #     assert tf_path is not None, (
        #         f"Transfer function for frequency {config.frequencies[f]} not provided."
        #     )
        #     if not use_new_reduced_TF:
        #         logger.info(f"Loading transfer function from {tf_path}")
        #         transfer = np.load(tf_path, allow_pickle=True)["full_tf"]
        #         inv_tf = get_reduced_TF(transfer)
        #     else:
        #         logger.warning("TESTING NEW METHOD FOR REDUCED TF REDUCTION")
        #         path_preprocessed_reduced_TF = manager.get_path_to_preprocessed_reduced_TF()
        #         logger.info(
        #             f"Loading preproc transfer function from {path_preprocessed_reduced_TF}"
        #         )
        #         reduced_TF_from_preproc = np.load(
        #             path_preprocessed_reduced_TF,
        #             allow_pickle=True,
        #         )
        #         inv_tf = np.abs(
        #             rebuilt_from_reduced_conj(reduced_TF_from_preproc["inv_sqrt_tf_bin_freq"][f]).T
        #         )
        #     noise_spectra_TF_corrected = np.einsum("lij,jl->il", inv_tf, noise_spectra_with_cross[f])
        #     noise_spectra_TF_corrected_unbined = nmt_bins.unbin_cell(noise_spectra_TF_corrected)

        #     output_noise_spectra_with_cross[f] = noise_spectra_TF_corrected
        #     output_noise_spectra_unbined_with_cross[f] = noise_spectra_TF_corrected_unbined
        dict_reconstructed_inv_TF = np.load(
            manager.get_path_to_reconstructed_inv_TF(), allow_pickle=True
        )
        inv_tf = dict_reconstructed_inv_TF[
            config.pre_proc_pars.DEBUGHARMONIC_noisepreproc_invTF_to_use
        ]

        # if use_new_reduced_TF:
        #     inv_tf = dict_reconstructed_inv_TF["inv_tf_new_reduce_freq"]
        # elif use_full_inv:
        #     inv_tf = dict_reconstructed_inv_TF["standard_inv_freq"]
        # else:
        #     inv_tf = dict_reconstructed_inv_TF["inv_tf_old_reduce_freq"]

        noise_spectra_with_cross = np.einsum("flij,fjl->fil", inv_tf, noise_spectra_with_cross)
        noise_spectra_unbined_with_cross = np.array(
            [
                nmt_bins.unbin_cell(noise_spectra_with_cross_f)
                for noise_spectra_with_cross_f in noise_spectra_with_cross
            ]
        )

    # TODO: remove bin_index_lminlmax?? we need to pad with 0 later in the compsep...

    nl_binned_with_cross = noise_spectra_with_cross
    nl_unbinned_with_cross = noise_spectra_unbined_with_cross[..., : config.lmax + 1]
    return nl_binned_with_cross, nl_unbinned_with_cross


def noise_preprocess_realisation(config: Config, manager: DataManager, id_sim: int | None) -> None:
    with Timer(f"noise-preproc-{id_sim}"):
        preprocessed = _preprocess_noise_maps(config, manager, id_sim)

    out_maps = manager.get_path_to_preprocessed_noise_maps(id_sim)
    logger.info(f"Saving pre-processed noise maps to {out_maps}")
    np.save(out_maps, preprocessed)

    if (
        config.noise_sim_pars.n_sim < config.map_sim_pars.n_sim
        and config.noise_sim_pars.DEBUG_save_TRUEnoise_simulations
    ):
        logger.warning(
            "noise_sim_pars.n_sim < map_sim_pars.n_sim but DEBUG_save_TRUEnoise_simulations is True. Only saving pre-processed TRUE noise maps for the first n_sim_noise realisations."
        )
        # TODO: check in config/manager and throw error if this is the case, to avoid confusion?

    if (
        config.noise_sim_pars.DEBUG_save_TRUEnoise_simulations
        and id_sim <= config.map_sim_pars.n_sim - 1
    ):
        preprocessed_TRUE_noise_maps = _preprocess_noise_maps(
            config, manager, id_sim, TRUE_noise_maps=True
        )
        out_TRUE_maps = manager.get_path_to_preprocessed_TRUE_noise_maps(id_sim)
        logger.info(f"Saving pre-processed TRUE noise maps to {out_TRUE_maps}")
        np.save(out_TRUE_maps, preprocessed_TRUE_noise_maps)

    if config.parametric_sep_pars.use_harmonic_compsep:
        if not config.pre_proc_pars.DEBUGHARMONIC_preproc_noise_TF_correction_alm:
            (
                nl_binned,
                nl_unbinned,
            ) = _harmonic_nl_contrib(config, manager, preprocessed)
        else:
            logger.warning("TESTING WITH ALMS FOR NOISE PREPROC")
            preproc_alm = _harmonic_alm_preproc_contrib(config, manager, id_sim)
            nl_binned, nl_unbinned = _harmonic_nl_from_alms_contrib(config, manager, preproc_alm)

            out_alms = manager.get_path_to_preprocessed_noise_alms(id_sim)
            logger.info(f"Saving pre-processed noise alms to {out_alms}")
            np.save(out_alms, preproc_alm)
        # nl_binned_with_cross, nl_unbinned_with_cross
        out_nl = manager.get_path_to_nl_noisecov_contrib(id_sim)
        out_nl_unbinned = manager.get_path_to_nl_noisecov_contrib_unbinned(id_sim)

        logger.info(f"Saving nl contribution to {out_nl}")
        np.save(out_nl, nl_binned)
        logger.info(f"Saving unbinned nl contribution to {out_nl_unbinned}")
        np.save(out_nl_unbinned, nl_unbinned)


def main():
    parser = argparse.ArgumentParser(description="Per-realisation noise preprocessing")
    parser.add_argument("--config", type=Path, required=True, help="config file")
    parser.add_argument("--sim", type=int, default=None, help="noise realisation index")
    args = parser.parse_args()

    config = Config.load_yaml(args.config)
    manager = DataManager(config)

    _world, rank, _size = get_world()
    if rank != 0:
        return

    manager.dump_config()
    manager.create_output_dirs(config.map_sim_pars.n_sim, config.noise_sim_pars.n_sim)

    n_sim_noise = config.noise_sim_pars.n_sim
    if args.sim is not None:
        id_sim = args.sim if n_sim_noise is not None else None
        noise_preprocess_realisation(config, manager, id_sim)
        return

    if n_sim_noise is None:
        noise_preprocess_realisation(config, manager, None)
        return

    for i in range(n_sim_noise):
        noise_preprocess_realisation(config, manager, i)
        logger.info(f"Finished noise preprocessing {i + 1} / {n_sim_noise}")


if __name__ == "__main__":
    main()
