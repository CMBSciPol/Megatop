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

from megatop import Config, DataManager
from megatop.utils import Timer, logger
from megatop.utils.binning import load_nmt_binning
from megatop.utils.mpi import get_world
from megatop.utils.preproc import common_beam_and_nside
from megatop.utils.spectra import initialize_nmt_workspace, spectra_from_namaster

HEALPY_DATA_PATH = os.getenv("HEALPY_LOCAL_DATA", None)


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


def _preprocess_noise_maps(config: Config, manager: DataManager, id_real: int | None) -> np.ndarray:
    noise_freq_maps = []
    for noise_filename in manager.get_noise_maps_filenames(id_real):
        logger.debug(f"Importing noise map: {noise_filename}")
        noise_freq_maps.append(hp.read_map(noise_filename, field=None).tolist())

    skip_beam = (
        np.all(np.array(config.pre_proc_pars.common_beam_correction) == np.array(config.beams))
        or config.pre_proc_pars.DEBUGskippreproc
    )
    if skip_beam:
        logger.info("Common beam correction is the same as the input beam, no need to apply it.")
        logger.warning("This is mostly for testing; may not represent the real noise.")
        return np.array(noise_freq_maps)

    return common_beam_and_nside(
        nside=config.nside,
        common_beam=config.pre_proc_pars.common_beam_correction,
        frequency_beams=config.beams,
        freq_maps=np.array(noise_freq_maps, dtype=object),
        lmax=config.lmax,
    )


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

    mask_analysis = hp.read_map(manager.path_to_analysis_mask)

    if config.parametric_sep_pars.harmonic_delta_ell != 1:
        with Timer("init-namaster-workspace"):
            workspaceff = initialize_nmt_workspace(
                nmt_bins=nmt_bins,
                analysis_mask=mask_analysis,
                beam=None,
                purify_e=False,
                purify_b=False,
                n_iter=10,
                lmax=config.lmax,
            )

        noise_spectra, noise_spectra_unbined = spectra_from_namaster(
            noise_freq_maps_preprocessed,
            mask_analysis,
            workspaceff,
            nmt_bins,
            compute_cross_freq=False,
            purify_e=False,
            purify_b=False,
            beam=None,
            return_all_spectra=config.pre_proc_pars.correct_for_TF,
        )

        if config.pre_proc_pars.correct_for_TF:
            logger.warning("Including transfer function in the pre-processed noise spectra.")
            output_noise_spectra = np.zeros([len(config.frequencies), 3, nmt_bins.get_n_bands()])
            output_noise_spectra_unbined = np.zeros(
                [len(config.frequencies), 3, noise_spectra_unbined.shape[-1]]
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

                logger.info(f"Loading transfer function from {tf_path}")
                transfer = np.load(tf_path, allow_pickle=True)["full_tf"]

                inv_tf = get_reduced_TF(transfer)

                noise_spectra_TF_corrected = np.einsum("lij,jl->il", inv_tf, noise_spectra[f])
                noise_spectra_TF_corrected_unbined = nmt_bins.unbin_cell(noise_spectra_TF_corrected)
                output_noise_spectra[f, 0] = noise_spectra_TF_corrected[0] * 0
                output_noise_spectra[f, 1] = noise_spectra_TF_corrected[0]
                output_noise_spectra[f, 2] = noise_spectra_TF_corrected[3]

                output_noise_spectra_unbined[f, 0] = noise_spectra_TF_corrected_unbined[0] * 0
                output_noise_spectra_unbined[f, 1] = noise_spectra_TF_corrected_unbined[0]
                output_noise_spectra_unbined[f, 2] = noise_spectra_TF_corrected_unbined[3]

            noise_spectra = output_noise_spectra
            noise_spectra_unbined = output_noise_spectra_unbined
    else:
        logger.warning(
            "Using harmonic delta ell = 1; healpy.anafast is used (not recommended for noise spectra)."
        )
        noise_spectra = np.array(
            [
                hp.anafast(noise_freq_maps_preprocessed[i], datapath=HEALPY_DATA_PATH)[:3]
                for i in range(len(config.frequencies))
            ]
        )
        noise_spectra_unbined = noise_spectra.copy()

    nl_binned = noise_spectra[..., bin_index_lminlmax]
    nl_unbinned = noise_spectra_unbined[..., ell_min:ell_max]
    return nl_binned, nl_unbinned


def noise_preprocess_realisation(config: Config, manager: DataManager, id_sim: int | None) -> None:
    with Timer(f"noise-preproc-{id_sim}"):
        preprocessed = _preprocess_noise_maps(config, manager, id_sim)

    out_maps = manager.get_path_to_preprocessed_noise_maps(id_sim)
    logger.info(f"Saving pre-processed noise maps to {out_maps}")
    np.save(out_maps, preprocessed)

    if config.parametric_sep_pars.use_harmonic_compsep:
        nl_binned, nl_unbinned = _harmonic_nl_contrib(config, manager, preprocessed)
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
