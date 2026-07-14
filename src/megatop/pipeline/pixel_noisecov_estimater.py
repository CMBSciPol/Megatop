"""Pixel (and harmonic) noise covariance aggregator.

Streams the per-realisation outputs produced by ``megatop-noise-preproc-run``
and averages them to form the final pixel covariance and (optionally) the
harmonic ``nl`` covariance. No MPI: realisation-level parallelism is now
handled by Snakemake fanning out one ``noise_preproc`` job per realisation.
"""

import argparse
from pathlib import Path

import healpy as hp
import numpy as np

from megatop import Config, DataManager
from megatop.utils import logger


def _iter_realisations(n_sim: int | None):
    """Yield ``id_sim`` indices for the per-realisation files.

    When ``n_sim`` is ``None`` (real-data mode), yield ``None`` once.
    """
    if n_sim is None:
        yield None
        return
    yield from range(n_sim)


def aggregate_noise_cov(manager: DataManager, config: Config) -> None:
    n_sim = config.noise_sim_pars.n_sim
    int_n_sim = 1 if n_sim is None else n_sim

    pixel_acc = np.zeros([len(config.frequencies), 3, hp.nside2npix(config.nside)])
    nl_acc = None
    nl_unbinned_acc = None
    use_harmonic = config.parametric_sep_pars.use_harmonic_compsep

    for id_sim in _iter_realisations(n_sim):
        maps_path = manager.get_path_to_preprocessed_noise_maps(id_sim)
        logger.info(f"Loading preprocessed noise maps from {maps_path}")
        maps = np.load(maps_path)
        pixel_acc += maps**2
        del maps

        if use_harmonic:
            nl = np.load(manager.get_path_to_nl_noisecov_contrib(id_sim))
            nl_un = np.load(manager.get_path_to_nl_noisecov_contrib_unbinned(id_sim))
            nl_acc = nl.copy() if nl_acc is None else nl_acc + nl
            nl_unbinned_acc = nl_un.copy() if nl_unbinned_acc is None else nl_unbinned_acc + nl_un

    pixel_mean = pixel_acc / int_n_sim
    np.save(manager.path_to_pixel_noisecov, pixel_mean)
    logger.info(f"Saved pixel noise covariance to {manager.path_to_pixel_noisecov}")

    if use_harmonic:
        np.save(manager.path_to_nl_noisecov, nl_acc / int_n_sim)
        np.save(manager.path_to_nl_noisecov_unbinned, nl_unbinned_acc / int_n_sim)
        logger.info(f"Saved harmonic nl covariance to {manager.path_to_nl_noisecov}")

    logger.info("\n\nNoise covariance matrix computation step completed successfully.\n\n")


def main():
    parser = argparse.ArgumentParser(description="Pixel noise covariance aggregator")
    parser.add_argument("--config", type=Path, required=True, help="config file")
    args = parser.parse_args()

    config = Config.load_yaml(args.config)
    manager = DataManager(config)
    manager.dump_config()
    manager.create_output_dirs(config.map_sim_pars.n_sim, config.noise_sim_pars.n_sim)

    aggregate_noise_cov(manager, config)


if __name__ == "__main__":
    main()
