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
from scipy.linalg import sqrtm

import megatop.utils.harmonic as hu
from megatop import Config, DataManager
from megatop.utils import Timer, logger
from megatop.utils.binning import load_nmt_binning
from megatop.utils.mpi import get_world
from megatop.utils.preproc import alm_common_beam, common_beam_and_nside, read_input_maps
from megatop.utils.spectra import initialize_nmt_workspace, spectra_from_namaster

HEALPY_DATA_PATH = os.getenv("HEALPY_LOCAL_DATA", None)


def rebuilt_from_reduced_conj(sqrt_corner_TF):
    shape_out = (4, 4, sqrt_corner_TF.shape[2])
    rebuilt_TF_manual = np.zeros(shape_out, dtype=complex)

    rebuilt_TF_manual[0, 0] = sqrt_corner_TF[0, 0] * np.conjugate(sqrt_corner_TF[0, 0])
    rebuilt_TF_manual[0, 1] = sqrt_corner_TF[0, 1] * np.conjugate(sqrt_corner_TF[0, 0])
    rebuilt_TF_manual[0, 2] = sqrt_corner_TF[0, 0] * np.conjugate(sqrt_corner_TF[0, 1])
    rebuilt_TF_manual[0, 3] = sqrt_corner_TF[0, 1] * np.conjugate(sqrt_corner_TF[0, 1])

    rebuilt_TF_manual[1, 0] = sqrt_corner_TF[0, 0] * np.conjugate(sqrt_corner_TF[1, 0])
    rebuilt_TF_manual[1, 1] = sqrt_corner_TF[0, 0] * np.conjugate(sqrt_corner_TF[1, 1])
    rebuilt_TF_manual[1, 2] = sqrt_corner_TF[0, 1] * np.conjugate(sqrt_corner_TF[1, 0])
    rebuilt_TF_manual[1, 3] = sqrt_corner_TF[0, 1] * np.conjugate(sqrt_corner_TF[1, 1])

    rebuilt_TF_manual[2, 0] = sqrt_corner_TF[1, 0] * np.conjugate(sqrt_corner_TF[0, 0])
    rebuilt_TF_manual[2, 1] = sqrt_corner_TF[1, 0] * np.conjugate(sqrt_corner_TF[0, 1])
    rebuilt_TF_manual[2, 2] = sqrt_corner_TF[1, 1] * np.conjugate(sqrt_corner_TF[0, 0])
    rebuilt_TF_manual[2, 3] = sqrt_corner_TF[1, 1] * np.conjugate(sqrt_corner_TF[0, 1])

    rebuilt_TF_manual[3, 0] = sqrt_corner_TF[1, 0] * np.conjugate(sqrt_corner_TF[1, 0])
    rebuilt_TF_manual[3, 1] = sqrt_corner_TF[1, 1] * np.conjugate(sqrt_corner_TF[1, 0])
    rebuilt_TF_manual[3, 2] = sqrt_corner_TF[1, 0] * np.conjugate(sqrt_corner_TF[1, 1])
    rebuilt_TF_manual[3, 3] = sqrt_corner_TF[1, 1] * np.conjugate(sqrt_corner_TF[1, 1])
    return rebuilt_TF_manual


def get_reduced_TF_for_Cl(inv_sqrt_tf_bin, transfer=None):
    if inv_sqrt_tf_bin is None:
        inv_sqrt_tf_full = np.linalg.inv([sqrtm(TF_ell.T) for TF_ell in transfer.T])[:, -4:, -4:]
        inv_sqrt_tf_bin = np.zeros((2, 2, inv_sqrt_tf_full.shape[0]), dtype=np.complex128)
        inv_sqrt_tf_bin[0, 0] = inv_sqrt_tf_full[:, 0, 0]
        inv_sqrt_tf_bin[0, 1] = inv_sqrt_tf_full[:, 1, 1]
        inv_sqrt_tf_bin[1, 0] = inv_sqrt_tf_full[:, 2, 2]
        inv_sqrt_tf_bin[1, 1] = inv_sqrt_tf_full[:, 3, 3]

    inv_tf_reduced = np.zeros((inv_sqrt_tf_bin.shape[-1], 4, 4), dtype=np.complex128)

    inv_tf_reduced[:, 0, 0] = inv_sqrt_tf_bin[0, 0] ** 2
    inv_tf_reduced[:, 0, 1] = inv_sqrt_tf_bin[0, 0] * inv_sqrt_tf_bin[0, 1]
    inv_tf_reduced[:, 0, 2] = inv_sqrt_tf_bin[0, 1] * inv_sqrt_tf_bin[0, 0]
    inv_tf_reduced[:, 0, 3] = inv_sqrt_tf_bin[0, 1] ** 2

    inv_tf_reduced[:, 1, 0] = inv_sqrt_tf_bin[0, 0] * inv_sqrt_tf_bin[1, 0]
    inv_tf_reduced[:, 1, 1] = inv_sqrt_tf_bin[0, 0] * inv_sqrt_tf_bin[1, 1]
    inv_tf_reduced[:, 1, 2] = inv_sqrt_tf_bin[1, 0] * inv_sqrt_tf_bin[0, 1]
    inv_tf_reduced[:, 1, 3] = inv_sqrt_tf_bin[0, 1] * inv_sqrt_tf_bin[1, 1]

    inv_tf_reduced[:, 2, 0] = inv_sqrt_tf_bin[1, 0] * inv_sqrt_tf_bin[0, 0]
    inv_tf_reduced[:, 2, 1] = inv_sqrt_tf_bin[0, 1] * inv_sqrt_tf_bin[1, 0]
    inv_tf_reduced[:, 2, 2] = inv_sqrt_tf_bin[1, 1] * inv_sqrt_tf_bin[0, 0]
    inv_tf_reduced[:, 2, 3] = inv_sqrt_tf_bin[1, 1] * inv_sqrt_tf_bin[0, 1]

    inv_tf_reduced[:, 3, 0] = inv_sqrt_tf_bin[1, 0] ** 2
    inv_tf_reduced[:, 3, 1] = inv_sqrt_tf_bin[1, 0] * inv_sqrt_tf_bin[1, 1]
    inv_tf_reduced[:, 3, 2] = inv_sqrt_tf_bin[1, 1] * inv_sqrt_tf_bin[1, 0]
    inv_tf_reduced[:, 3, 3] = inv_sqrt_tf_bin[1, 1] ** 2

    return np.real(inv_tf_reduced)


# def pixel_noisecov_estimation(manager: DataManager, config: Config):
#     comm, rank, size = get_world()
#     root = 0
#     MemoryUsage(f"rank = {rank} ")

#     logger.info(f"rank = {rank}, size = {size}")
#     noise_cov_preprocessed = np.zeros([len(config.frequencies), 3, hp.nside2npix(config.nside)])
#     test_alm_TF_noise = True
#     if test_alm_TF_noise:
#         noise_cov_alm_preprocessed = np.zeros(
#             [
#                 len(config.frequencies),
#                 2,
#                 hp.Alm.getsize(config.parametric_sep_pars.harmonic_lmax - 1),
#             ]
#         )

#     if config.parametric_sep_pars.use_harmonic_compsep:
#         nmt_bins = load_nmt_binning(manager)
#         bin_index_lminlmax = np.load(manager.path_to_binning, allow_pickle=True)[
#             "bin_index_lminlmax"
#         ]

#         ell_min_namaster = config.parametric_sep_pars.harmonic_lmin
#         ell_max_namaster = config.parametric_sep_pars.harmonic_lmax

#         mask_analysis = hp.read_map(manager.path_to_analysis_mask)

#         with Timer("init-namaster-workspace"):
#             workspaceff = initialize_nmt_workspace(
#                 nmt_bins,
#                 manager.path_to_lensed_scalar,
#                 config.nside,
#                 mask_analysis,
#                 effective_beam=None,
#                 purify_e=config.map2cl_pars.purify_e,
#                 purify_b=config.map2cl_pars.purify_b,
#                 n_iter=10,
#             )

#         ell_total = len(nmt_bins.get_effective_ells()[bin_index_lminlmax])

#         # Initializeing the noise spectra
#         cl_noise_cov_preprocessed_unbinned = np.zeros(
#             (len(config.frequencies), 3, ell_max_namaster - ell_min_namaster)
#         )

#         cl_noise_cov_preprocessed = np.zeros((len(config.frequencies), 3, ell_total))

#         cl_noise_cov_preprocessed_unbinned_with_cross = np.zeros(
#             (len(config.frequencies), 5, ell_max_namaster - ell_min_namaster)
#         )

#         cl_noise_cov_preprocessed_with_cross = np.zeros((len(config.frequencies), 5, ell_total))

#     # Importing noise maps


def get_reduced_TF(transfer):
    inv_sqrt_tf_full = np.linalg.inv([sqrtm(TF_ell.T) for TF_ell in transfer.T])[:, -4:, -4:]
    inv_sqrt_tf_bin = np.zeros((2, 2, inv_sqrt_tf_full.shape[0]), dtype=np.complex128)
    inv_sqrt_tf_bin[0, 0] = inv_sqrt_tf_full[:, 0, 0]
    inv_sqrt_tf_bin[0, 1] = inv_sqrt_tf_full[:, 1, 1]
    inv_sqrt_tf_bin[1, 0] = inv_sqrt_tf_full[:, 2, 2]
    inv_sqrt_tf_bin[1, 1] = inv_sqrt_tf_full[:, 3, 3]

    inv_tf_reduced = np.zeros((inv_sqrt_tf_full.shape[0], 4, 4), dtype=np.complex128)

    inv_tf_reduced[:, 0, 0] = inv_sqrt_tf_bin[0, 0] ** 2
    inv_tf_reduced[:, 0, 1] = inv_sqrt_tf_bin[0, 0] * inv_sqrt_tf_bin[0, 1]
    inv_tf_reduced[:, 0, 2] = inv_sqrt_tf_bin[0, 1] * inv_sqrt_tf_bin[0, 0]
    inv_tf_reduced[:, 0, 3] = inv_sqrt_tf_bin[0, 1] ** 2

    inv_tf_reduced[:, 1, 0] = inv_sqrt_tf_bin[0, 0] * inv_sqrt_tf_bin[1, 0]
    inv_tf_reduced[:, 1, 1] = inv_sqrt_tf_bin[0, 0] * inv_sqrt_tf_bin[1, 1]
    inv_tf_reduced[:, 1, 2] = inv_sqrt_tf_bin[1, 0] * inv_sqrt_tf_bin[0, 1]
    inv_tf_reduced[:, 1, 3] = inv_sqrt_tf_bin[0, 1] * inv_sqrt_tf_bin[1, 1]

    inv_tf_reduced[:, 2, 0] = inv_sqrt_tf_bin[1, 0] * inv_sqrt_tf_bin[0, 0]
    inv_tf_reduced[:, 2, 1] = inv_sqrt_tf_bin[0, 1] * inv_sqrt_tf_bin[1, 0]
    inv_tf_reduced[:, 2, 2] = inv_sqrt_tf_bin[1, 1] * inv_sqrt_tf_bin[0, 0]
    inv_tf_reduced[:, 2, 3] = inv_sqrt_tf_bin[1, 1] * inv_sqrt_tf_bin[0, 1]

    inv_tf_reduced[:, 3, 0] = inv_sqrt_tf_bin[1, 0] ** 2
    inv_tf_reduced[:, 3, 1] = inv_sqrt_tf_bin[1, 0] * inv_sqrt_tf_bin[1, 1]
    inv_tf_reduced[:, 3, 2] = inv_sqrt_tf_bin[1, 1] * inv_sqrt_tf_bin[1, 0]
    inv_tf_reduced[:, 3, 3] = inv_sqrt_tf_bin[1, 1] ** 2

    return np.real(inv_tf_reduced)


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
    input_maps = read_input_maps(manager.get_maps_filenames(id_real))
    logger.info(
        f"Input maps have shapes: {[input_maps[i].shape for i in range(len(config.frequencies))]}"
    )
    freq_beams = config.beams
    common_beam = config.pre_proc_pars.common_beam_correction

    if config.pre_proc_pars.DEBUGHARMONICuse_namaster_alms:
        analysis_mask = hp.read_map(manager.path_to_analysis_mask)
        mask_alm_computation = analysis_mask
    else:
        binary_mask = hp.read_map(manager.path_to_binary_mask)
        mask_alm_computation = binary_mask

    freq_alms_convolved = alm_common_beam(
        nside=config.nside,
        common_beam=common_beam,
        frequency_beams=freq_beams,
        freq_maps=np.array(input_maps),
        lmax=config.lmax,
        analysis_mask=mask_alm_computation,
        harmonic_analysis_lmax=config.parametric_sep_pars.harmonic_lmax,
        purify_e=config.map2cl_pars.purify_e,
        purify_b=config.map2cl_pars.purify_b,
        use_namaster_alms=config.pre_proc_pars.DEBUGHARMONICuse_namaster_alms,
    )

    for f, tf_path in enumerate(manager.get_TF_filenames()):
        logger.warning("TESTING NEW METHOD FOR REDUCED TF REDUCTION")
        path_preprocessed_reduced_TF = manager.get_path_to_preprocessed_reduced_TF()
        logger.info(f"Loading preproc transfer function from {path_preprocessed_reduced_TF}")
        reduced_TF_from_preproc = np.load(
            path_preprocessed_reduced_TF,
            allow_pickle=True,
        )
        inv_sqrt_tf_lm = reduced_TF_from_preproc["inv_sqrt_tf_lm_freq"][f]

        freq_alms_convolved[f] = np.einsum("ijl,jl->il", inv_sqrt_tf_lm, freq_alms_convolved[f])

    return


def _harmonic_nl_contrib(
    config: Config,
    manager: DataManager,
    noise_freq_maps_preprocessed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the harmonic ``nl`` contribution (binned + unbinned) for one realisation."""
    nmt_bins = load_nmt_binning(manager)
    bin_index_lminlmax = np.load(manager.path_to_binning, allow_pickle=True)["bin_index_lminlmax"]

    ell_min = config.parametric_sep_pars.harmonic_lmin
    ell_max = config.parametric_sep_pars.harmonic_lmax

    mask_analysis = hp.read_map(manager.path_to_analysis_mask, dtype=np.float64)

    if config.parametric_sep_pars.harmonic_delta_ell != 1:
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

        noise_spectra, noise_spectra_unbined = spectra_from_namaster(
            noise_freq_maps_preprocessed,
            mask_analysis,
            workspaceff,
            nmt_bins,
            compute_cross_freq=False,
            purify_e=config.map2cl_pars.purify_e,
            purify_b=config.map2cl_pars.purify_b,
            beam=None,
            return_all_spectra=config.pre_proc_pars.correct_for_TF,
            lmax=config.lmax,
        )
        noise_spectra_with_cross = noise_spectra.copy()
        noise_spectra_unbined_with_cross = noise_spectra_unbined.copy()

        if config.pre_proc_pars.correct_for_TF:
            logger.warning("Including transfer function in the pre-processed noise spectra.")
            output_noise_spectra = np.zeros([len(config.frequencies), 3, nmt_bins.get_n_bands()])
            output_noise_spectra_unbined = np.zeros(
                [len(config.frequencies), 3, noise_spectra_unbined.shape[-1]]
            )

            # With Cross include EB and BE cross spectra
            output_noise_spectra_with_cross = np.zeros(
                [len(config.frequencies), 5, nmt_bins.get_n_bands()]
            )
            output_noise_spectra_unbined_with_cross = np.zeros(
                [len(config.frequencies), 5, noise_spectra_unbined.shape[-1]]
            )

            for f, tf_path in enumerate(manager.get_TF_filenames()):
                if tf_path is None:
                    logger.warning(
                        f"Transfer function for frequency {config.frequencies[f]} not provided, skipping."
                    )
                    output_noise_spectra[f, 0] = noise_spectra[f, 0] * 0
                    output_noise_spectra[f, 1] = noise_spectra[f, 0]
                    output_noise_spectra[f, 2] = noise_spectra[f, 3]
                    output_noise_spectra_unbined[f, 0] = noise_spectra_unbined[f, 0] * 0
                    output_noise_spectra_unbined[f, 1] = noise_spectra_unbined[f, 0]
                    output_noise_spectra_unbined[f, 2] = noise_spectra_unbined[f, 3]
                    continue

                use_new_reduced_TF = True
                if not use_new_reduced_TF:
                    logger.info(f"Loading transfer function from {tf_path}")
                    transfer = np.load(tf_path, allow_pickle=True)["full_tf"]
                    inv_tf = get_reduced_TF(transfer)
                else:
                    logger.warning("TESTING NEW METHOD FOR REDUCED TF REDUCTION")
                    path_preprocessed_reduced_TF = manager.get_path_to_preprocessed_reduced_TF()
                    logger.info(
                        f"Loading preproc transfer function from {path_preprocessed_reduced_TF}"
                    )
                    reduced_TF_from_preproc = np.load(
                        path_preprocessed_reduced_TF,
                        allow_pickle=True,
                    )
                    inv_tf = np.abs(
                        rebuilt_from_reduced_conj(
                            reduced_TF_from_preproc["inv_sqrt_tf_bin_freq"][f]
                        ).T
                    )
                noise_spectra_TF_corrected = np.einsum("lij,jl->il", inv_tf, noise_spectra[f])
                noise_spectra_TF_corrected_unbined = nmt_bins.unbin_cell(noise_spectra_TF_corrected)
                output_noise_spectra[f, 0] = noise_spectra_TF_corrected[0] * 0
                output_noise_spectra[f, 1] = noise_spectra_TF_corrected[0]
                output_noise_spectra[f, 2] = noise_spectra_TF_corrected[3]

                output_noise_spectra_unbined[f, 0] = noise_spectra_TF_corrected_unbined[0] * 0
                output_noise_spectra_unbined[f, 1] = noise_spectra_TF_corrected_unbined[0]
                output_noise_spectra_unbined[f, 2] = noise_spectra_TF_corrected_unbined[3]

                output_noise_spectra_with_cross[f, 0] = noise_spectra_TF_corrected[0] * 0
                output_noise_spectra_with_cross[f, 1] = noise_spectra_TF_corrected[0]
                output_noise_spectra_with_cross[f, 2] = noise_spectra_TF_corrected[1]
                output_noise_spectra_with_cross[f, 3] = noise_spectra_TF_corrected[2]
                output_noise_spectra_with_cross[f, 4] = noise_spectra_TF_corrected[3]

                output_noise_spectra_unbined_with_cross[f, 0] = (
                    noise_spectra_TF_corrected_unbined[0] * 0
                )
                output_noise_spectra_unbined_with_cross[f, 1] = noise_spectra_TF_corrected_unbined[
                    0
                ]
                output_noise_spectra_unbined_with_cross[f, 2] = noise_spectra_TF_corrected_unbined[
                    1
                ]
                output_noise_spectra_unbined_with_cross[f, 3] = noise_spectra_TF_corrected_unbined[
                    2
                ]
                output_noise_spectra_unbined_with_cross[f, 4] = noise_spectra_TF_corrected_unbined[
                    3
                ]

            noise_spectra = output_noise_spectra
            noise_spectra_unbined = output_noise_spectra_unbined
            noise_spectra_with_cross = output_noise_spectra_with_cross
            noise_spectra_unbined_with_cross = output_noise_spectra_unbined_with_cross

    else:
        logger.warning(
            "Using harmonic delta ell = 1; healpy.anafast is used (not recommended for noise spectra)."
        )
        noise_spectra = np.array(
            [
                hu.anafast(noise_freq_maps_preprocessed[i])[:3]
                for i in range(len(config.frequencies))
            ]
        )
        noise_spectra_unbined = noise_spectra.copy()

    nl_binned = noise_spectra[..., bin_index_lminlmax]
    nl_unbinned = noise_spectra_unbined[..., ell_min : ell_max + 1]
    nl_binned_with_cross = noise_spectra_with_cross[..., bin_index_lminlmax]
    nl_unbinned_with_cross = noise_spectra_unbined_with_cross[..., ell_min : ell_max + 1]
    return nl_binned, nl_unbinned, nl_binned_with_cross, nl_unbinned_with_cross


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
        nl_binned, nl_unbinned, nl_binned_with_cross, nl_unbinned_with_cross = _harmonic_nl_contrib(
            config, manager, preprocessed
        )

        out_nl = manager.get_path_to_nl_noisecov_contrib(id_sim)
        out_nl_unbinned = manager.get_path_to_nl_noisecov_contrib_unbinned(id_sim)

        out_nl_with_cross = manager.get_path_to_nl_noisecov_contrib_with_cross(id_sim)
        out_nl_unbinned_with_cross = manager.get_path_to_nl_noisecov_contrib_unbinned_with_cross(
            id_sim
        )
        logger.info(f"Saving nl contribution to {out_nl}")
        np.save(out_nl, nl_binned)
        logger.info(f"Saving unbinned nl contribution to {out_nl_unbinned}")
        np.save(out_nl_unbinned, nl_unbinned)
        logger.info(f"Saving nl with cross contribution to {out_nl_with_cross}")
        np.save(out_nl_with_cross, nl_binned_with_cross)
        logger.info(f"Saving unbinned nl with cross contribution to {out_nl_unbinned_with_cross}")
        np.save(out_nl_unbinned_with_cross, nl_unbinned_with_cross)


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
