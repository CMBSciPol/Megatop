import argparse
import tracemalloc
from pathlib import Path

import healpy as hp
import numpy as np

from megatop import Config, DataManager
from megatop.utils import logger
from megatop.utils.mpi import MPISUM
from megatop.utils.preproc import common_beam_and_nside
from megatop.utils.utils import MemoryUsage


def pixel_noisecov_estimation(manager: DataManager, config: Config):
    tracemalloc.start()

    try:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.rank
        root = 0

    except ImportError:
        logger.info("Could not find MPI. Proceeding without.")

        comm = None
        root = 0
        rank = 0
        size = 1

    MemoryUsage(f"rank = {rank} ")

    logger.info(f"rank = {rank}, size = {size}")
    noise_cov_preprocessed = np.zeros([len(config.frequencies), 3, hp.nside2npix(config.nside)])

    # Importing noise maps
    nreal = config.noise_cov_pars.nrealizations

    # The None case of nreal is useful when calling get_noise_map_filename
    # so we need to handle it when creating the list of realisations to loop over
    int_nreal = 1 if nreal is None else nreal
    realisation_list = np.arange(int_nreal)

    # splitting the list of simulation between the ranks of the process:
    rank_realisation_list = np.array_split(realisation_list, size)[rank]

    for id_realisation in rank_realisation_list:
        noise_freq_maps = []

        id_real = None if nreal is None else id_realisation

        logger.info(f"id_realisation = {id_real}")
        logger.info(f"in = {rank_realisation_list}")  # debug in logger

        for noise_filename in manager.get_noise_maps_filenames(id_real):
            logger.debug(f"Importing noise map: {noise_filename}")
            noise_freq_maps.append(hp.read_map(noise_filename, field=None).tolist())

        if np.all(np.array(config.pre_proc_pars.common_beam_correction) == np.array(config.beams)):
            logger.info(
                "Common beam correction is the same as the input beam, no need to apply it."
            )
            logger.info(
                "WARNING: this is mostly for testing it might not actually represent the real noise"
            )
            # freq_noise_maps_array = np.array(
            #     freq_noise_maps_array
            # )  # not using dtype=object to avoid issue with addition for noise_cov_preprocessed
            noise_freq_maps_preprocessed = noise_freq_maps

        else:
            noise_freq_maps = np.array(noise_freq_maps, dtype=object)
            noise_freq_maps_preprocessed = common_beam_and_nside(
                nside=config.nside,
                common_beam=config.pre_proc_pars.common_beam_correction,
                frequency_beams=config.beams,
                freq_maps=noise_freq_maps,
            )

        if config.noise_cov_pars.save_preprocessed_noise_maps:
            manager.get_path_to_preprocessed_noise_maps(sub=id_real).parent.mkdir(
                exist_ok=True, parents=True
            )
            logger.info("Saving pre-processed noise maps to disk")
            np.save(
                manager.get_path_to_preprocessed_noise_maps(sub=id_real),
                noise_freq_maps_preprocessed,
            )

        MemoryUsage(f"memory for id_realisation = {id_real} ")

        noise_cov_preprocessed += noise_freq_maps_preprocessed**2

    if comm is not None:
        noise_cov_preprocessed_recvbuf = MPISUM(noise_cov_preprocessed, comm, rank, root)
    else:
        noise_cov_preprocessed_recvbuf = noise_cov_preprocessed

    if rank == root:
        # Average noise_cov and noise_cov_preprocessed over nsims
        noise_cov_preprocessed_mean = noise_cov_preprocessed_recvbuf / int_nreal
    else:
        noise_cov_preprocessed_mean = None

    if rank == root:
        manager.path_to_covar.mkdir(exist_ok=True, parents=True)
        np.save(manager.path_to_pixel_noisecov, noise_cov_preprocessed_mean)

    if rank == root:
        logger.info("\n\nNoise covariance matrix computation step completed successfully.\n\n")


def main():
    parser = argparse.ArgumentParser(description="Pixel noise covariance estimater")
    parser.add_argument("--config", type=Path, help="config file")
    args = parser.parse_args()
    if args.config is None:
        logger.warning("No config file provided, using example config")
        config = Config.get_example()
    else:
        config = Config.from_yaml(args.config)
    manager = DataManager(config)
    manager.dump_config()
    pixel_noisecov_estimation(manager, config)


if __name__ == "__main__":
    main()
