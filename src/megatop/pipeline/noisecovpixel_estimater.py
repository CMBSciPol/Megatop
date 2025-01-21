import argparse
import os
import tracemalloc

import healpy as hp
import numpy as np
from mpi4py import MPI

from megatop.utils import BBmeta
from megatop.utils.preproc_utils import CommonBeamConvAndNsideModification
from megatop.utils.utils import MPISUM, MemoryUsage


# =================================================================================
# =                     Main function, calling the wrappers etc                   =
# =================================================================================


def GetNoiseCov(meta):
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

        pass

    MemoryUsage(meta, f"rank = {rank} ")

    meta.logger.info(f"rank = {rank}, size = {size}")
   
    meta.logger.info("rank = {}, size = {}".format(rank, size))

    noise_cov_preprocessed = np.zeros(
        [len(meta.frequencies), 3, hp.nside2npix(meta.general_pars["nside"])]
    )

    # Importing noise maps
    maps_list = meta.maps_list
    nside_in_list = []

    nreal = meta.noise_cov_pars["nrealisation_noise_cov"]

    # The None case of nreal is useful when calling get_noise_map_filename
    # so we need to handle it when creating the list of realisations to loop over
    if nreal is None:
        int_nreal = 1
    else:
        int_nreal = nreal
    realisation_list = np.arange(int_nreal)

    # splitting the list of simulation between the ranks of the process:
    rank_realisation_list = np.array_split(realisation_list, size)[rank]

    for id_realisation in rank_realisation_list:
        freq_noise_maps_array = []

        if nreal is None:
            id_realisation = None

        meta.logger.info(f"id_realisation = {id_realisation}")
        meta.logger.info(f"in = {rank_realisation_list}")  # debug in logger

        for m in maps_list:
            path_noise_map = meta.get_noise_map_filename(m, id_sim=id_realisation)

            meta.logger.info(f"Importing noise map: {path_noise_map}")

            freq_noise_maps_array.append(hp.read_map(path_noise_map, field=None).tolist())
            nside_in_list.append(hp.get_nside(freq_noise_maps_array[-1][-1]))

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
            freq_noise_maps_array = np.array(
                freq_noise_maps_array
            )  # not using dtype=object to avoid issue with addition for noise_cov_preprocessed
            freq_noise_maps_pre_processed = freq_noise_maps_array

        else:
            freq_noise_maps_array = np.array(freq_noise_maps_array, dtype=object)
            freq_noise_maps_pre_processed = CommonBeamConvAndNsideModification(
                meta, freq_noise_maps_array
            )

        if meta.noise_cov_pars.get("save_preprocessed_noise_maps"):
            meta.logger.info("Saving pre-processed noise maps to disk")

            if nreal is not None:
                add_sim_id = f"_{id_realisation:04d}.npy"
            else:
                add_sim_id = ".npy"
            np.save(
                os.path.join(meta.covmat_directory, "freq_noise_maps_preprocessed" + add_sim_id),
                freq_noise_maps_pre_processed,
            )

        MemoryUsage(meta, f"memory for id_realisation = {id_realisation} ")

        noise_cov_preprocessed += freq_noise_maps_pre_processed**2

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="simplistic simulator")
    parser.add_argument("--globals", type=str, help="Path to yaml with global parameters")

    args = parser.parse_args()
    meta = BBmeta(args.globals)

    GetNoiseCov(meta)
