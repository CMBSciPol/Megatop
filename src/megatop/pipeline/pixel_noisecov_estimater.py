import argparse
import os
import tracemalloc

import healpy as hp
import numpy as np
from mpi4py import MPI

from megatop.utils import BBmeta
from megatop.utils.mpi import MPISUM
from megatop.utils.preproc_utils import common_beam_and_nside
from megatop.utils.utils import MemoryUsage

# =================================================================================
# =                     Main function, calling the wrappers etc                   =
# =================================================================================


def pixel_noisecov_estimation(meta):
    """
    Estimating the noise covariance matrix from noise map(s) saved on disk.
    Directly saves the noise covariance matrix on disk.

    Parameters
    ----------
    meta : object
        The metadata manager object from BBmeta.

    Returns
    -------
    None

    """

    tracemalloc.start()

    try:
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.rank
        root = 0

    except (ModuleNotFoundError, ImportError) as e:
        # Error handling
        meta.logger.info(f"ERROR IN MPI:{e}")
        meta.logger.info("Proceeding without MPI\n")

        root = 0
        rank = 0
        size = 1

    MemoryUsage(meta, f"rank = {rank} ")

    meta.logger.info(f"rank = {rank}, size = {size}")
    noise_cov_preprocessed = np.zeros([len(meta.frequencies), 3, hp.nside2npix(meta.nside)])

    # Importing noise maps
    maps_list = meta.maps_list
    nside_in_list = []

    nreal = meta.noise_cov_pars["nrealisation_noise_cov"]

    # The None case of nreal is useful when calling get_noise_map_filename
    # so we need to handle it when creating the list of realisations to loop over
    int_nreal = 1 if nreal is None else nreal
    realisation_list = np.arange(int_nreal)

    # splitting the list of simulation between the ranks of the process:
    rank_realisation_list = np.array_split(realisation_list, size)[rank]

    for id_realisation in rank_realisation_list:
        noise_freq_maps = []

        id_real = None if nreal is None else id_realisation

        meta.logger.info(f"id_realisation = {id_real}")
        meta.logger.info(f"in = {rank_realisation_list}")  # debug in logger

        for m in maps_list:
            path_noise_map = meta.get_noise_map_filename(m, id_sim=id_real)

            meta.logger.info(f"Importing noise map: {path_noise_map}")

            noise_freq_maps.append(hp.read_map(path_noise_map, field=None).tolist())
            nside_in_list.append(hp.get_nside(noise_freq_maps[-1][-1]))

        if np.all(
            np.array(meta.pre_proc_pars["common_beam_correction"])
            == np.array(meta.pre_proc_pars["fwhm"])
        ):
            meta.logger.info(
                "Common beam correction is the same as the input beam, no need to apply it."
            )
            meta.logger.info(
                "WARNING: this is mostly for testing it might not actually represent the real noise"
            )
            # freq_noise_maps_array = np.array(
            #     freq_noise_maps_array
            # )  # not using dtype=object to avoid issue with addition for noise_cov_preprocessed
            noise_freq_maps_preprocessed = noise_freq_maps

        else:
            noise_freq_maps = np.array(noise_freq_maps, dtype=object)
            noise_freq_maps_preprocessed = common_beam_and_nside(meta, noise_freq_maps)

        if meta.noise_cov_pars.get("save_preprocessed_noise_maps"):
            meta.logger.info("Saving pre-processed noise maps to disk")

            add_sim_id = f"_{id_real:04d}.npy" if nreal is not None else ".npy"
            np.save(
                os.path.join(meta.covmat_directory, "freq_noise_maps_preprocessed" + add_sim_id),
                noise_freq_maps_preprocessed,
            )

        MemoryUsage(meta, f"memory for id_realisation = {id_real} ")

        noise_cov_preprocessed += noise_freq_maps_preprocessed**2

    noise_cov_preprocessed_recvbuf = MPISUM(noise_cov_preprocessed, comm, rank, root)

    if rank == root:
        # Average noise_cov and noise_cov_preprocessed over nsims
        noise_cov_preprocessed_mean = noise_cov_preprocessed_recvbuf / int_nreal
    else:
        noise_cov_preprocessed_mean = None

    if rank == root:
        np.save(
            os.path.join(meta.covmat_directory, "pixel_noise_cov_preprocessed.npy"),
            noise_cov_preprocessed_mean,
        )

    if rank == root:
        meta.logger.info("\n\nNoise covariance matrix computation step completed successfully.\n\n")


# ==================================================================================================
# =                                           MAIN CALL                                            =
# ==================================================================================================


def main():
    parser = argparse.ArgumentParser(description="simplistic simulator")
    parser.add_argument("--globals", type=str, help="Path to yaml with global parameters")

    args = parser.parse_args()
    meta = BBmeta(args.globals)

    pixel_noisecov_estimation(meta)


if __name__ == "__main__":
    main()
