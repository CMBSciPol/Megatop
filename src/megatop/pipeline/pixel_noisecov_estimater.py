import argparse
import tracemalloc
from pathlib import Path

import healpy as hp
import numpy as np
import pymaster as nmt

from megatop import Config, DataManager
from megatop.utils import Timer, logger
from megatop.utils.mpi import MPISUM, get_world
from megatop.utils.preproc import common_beam_and_nside
from megatop.utils.spectra import create_binning, initialize_nmt_workspace
from megatop.utils.utils import MemoryUsage


def spectra_from_namaster(
    freq_noise_maps,
    mask_analysis,
    workspaceff,
    nmt_bins,
    compute_cross_freq=False,
    purify_e=False,
    purify_b=False,
):
    # TODO: put in utils
    # TODO: write docstring
    if compute_cross_freq:
        msg = "Cross-frequency spectra computation is not implemented yet"
        raise NotImplementedError(msg)

    cl_decoupled_freq = []
    unbin_cl_decoupled_freq = []
    for f in range(freq_noise_maps.shape[0]):
        fields = nmt.NmtField(
            mask_analysis,
            freq_noise_maps[f, 1:],
            beam=None,
            purify_e=purify_e,
            purify_b=purify_b,
            n_iter=10,
        )
        cl_coupled = nmt.compute_coupled_cell(fields, fields)
        cl_decoupled = workspaceff.decouple_cell(cl_coupled)
        unbin_cl_decoupled = nmt_bins.unbin_cell(cl_decoupled)

        # Keeping only the T, E, B components, setting T to zero
        # Warning: we are ignoring the EB cross-spectra here
        cl_decoupled_freq.append([cl_decoupled[0] * 0, cl_decoupled[0], cl_decoupled[3]])
        unbin_cl_decoupled_freq.append(
            [unbin_cl_decoupled[0] * 0, unbin_cl_decoupled[0], unbin_cl_decoupled[3]]
        )

    return np.array(cl_decoupled_freq), np.array(unbin_cl_decoupled_freq)


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

    if config.parametric_sep_pars.use_harmonic_compsep:
        # Initilizing the binning sccheme used in the harmonic component separation in namaster
        bin_low, bin_high, bin_centre = create_binning(
            config.nside,
            config.parametric_sep_pars.harmonic_delta_ell,
            end_first_bin=config.parametric_sep_pars.harmonic_delta_ell,
        )
        nmt_bins = nmt.NmtBin.from_edges(bin_low, bin_high + 1)

        ell_min_namaster = config.parametric_sep_pars.harmonic_lmin
        ell_max_namaster = config.parametric_sep_pars.harmonic_lmax
        bin_index_lminlmax = np.where(
            (bin_low >= ell_min_namaster) & (bin_high <= ell_max_namaster)
        )[0]

        # Bins from Carlos BBMASTER paper:
        USE_BBMASTER_BINS = True
        if USE_BBMASTER_BINS:
            logger.warning("Using EXTERNAL BBMASTER bins for the harmonic component separation.")

            nmt_bins = nmt.NmtBin.from_nside_linear(config.nside, nlb=10, is_Dell=False)
            bin_index_lminlmax = np.where(
                (nmt_bins.get_effective_ells() >= ell_min_namaster)
                & (nmt_bins.get_effective_ells() <= ell_max_namaster)
            )[0]

        mask_analysis = hp.read_map(manager.path_to_analysis_mask)

        with Timer("init-namaster-workspace"):
            workspaceff = initialize_nmt_workspace(
                nmt_bins,
                manager.path_to_lensed_scalar,
                config.nside,
                mask_analysis,
                effective_beam=None,
                purify_e=False,
                purify_b=False,
                n_iter=10,
            )

        ell_total = len(nmt_bins.get_effective_ells()[bin_index_lminlmax])

        # Initializeing the noise spectra
        cl_noise_cov_preprocessed_unbinned = np.zeros(
            (len(config.frequencies), 3, ell_max_namaster - ell_min_namaster)
        )

        cl_noise_cov_preprocessed = np.zeros((len(config.frequencies), 3, ell_total))

    # Importing noise maps
    n_sim = config.noise_sim_pars.n_sim

    # The None case of n_sim is useful when calling get_noise_map_filename
    # so we need to handle it when creating the list of realisations to loop over
    int_n_sim = 1 if n_sim is None else n_sim
    realisation_list = np.arange(int_n_sim)

    # splitting the list of simulation between the ranks of the process:
    rank_realisation_list = np.array_split(realisation_list, size)[rank]

    for id_realisation in rank_realisation_list:
        noise_freq_maps = []

        id_real = None if n_sim is None else id_realisation

        logger.info(f"Noise realisation {id_real + 1}/{n_sim}")
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
            noise_freq_maps_preprocessed = np.array(noise_freq_maps)

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

        MemoryUsage(f"Memory for noise realisation {id_real + 1}: ")

        noise_cov_preprocessed += noise_freq_maps_preprocessed**2

        if config.parametric_sep_pars.use_harmonic_compsep:
            # Computing the noise spectra from the preprocessed noise maps using namaster
            noise_spectra, noise_spectra_unbined = spectra_from_namaster(
                noise_freq_maps_preprocessed,
                mask_analysis,
                workspaceff,
                nmt_bins,
                compute_cross_freq=False,
            )
            noise_spectra = noise_spectra[..., bin_index_lminlmax]

            # Adding the noise spectra to the ones from previous realisations
            cl_noise_cov_preprocessed += noise_spectra
            cl_noise_cov_preprocessed_unbinned += noise_spectra_unbined[
                ..., ell_min_namaster:ell_max_namaster
            ]

    if comm is not None:
        noise_cov_preprocessed_recvbuf = MPISUM(noise_cov_preprocessed, comm, rank, root)
        if config.parametric_sep_pars.use_harmonic_compsep:
            noise_cov_preprocessed_recvbuf_cl = MPISUM(cl_noise_cov_preprocessed, comm, rank, root)
            noise_cov_preprocessed_recvbuf_cl_unbinned = MPISUM(
                cl_noise_cov_preprocessed_unbinned, comm, rank, root
            )
    else:
        noise_cov_preprocessed_recvbuf = noise_cov_preprocessed
        if config.parametric_sep_pars.use_harmonic_compsep:
            noise_cov_preprocessed_recvbuf_cl = cl_noise_cov_preprocessed
            noise_cov_preprocessed_recvbuf_cl_unbinned = cl_noise_cov_preprocessed_unbinned

    if rank == root:
        # Average noise_cov and noise_cov_preprocessed over nsims
        noise_cov_preprocessed_mean = noise_cov_preprocessed_recvbuf / int_n_sim
        if config.parametric_sep_pars.use_harmonic_compsep:
            noise_cov_preprocessed_mean_cl = noise_cov_preprocessed_recvbuf_cl / int_n_sim
            noise_cov_preprocessed_recvbuf_cl_unbinned = (
                noise_cov_preprocessed_recvbuf_cl_unbinned / int_n_sim
            )

    else:
        noise_cov_preprocessed_mean = None
        if config.parametric_sep_pars.use_harmonic_compsep:
            noise_cov_preprocessed_mean_cl = None
            noise_cov_preprocessed_recvbuf_cl_unbinned = None

    if rank == root:
        manager.path_to_covar.mkdir(exist_ok=True, parents=True)
        np.save(manager.path_to_pixel_noisecov, noise_cov_preprocessed_mean)
        if config.parametric_sep_pars.use_harmonic_compsep:
            np.save(manager.path_to_nl_noisecov, noise_cov_preprocessed_mean_cl)
            np.save(
                manager.path_to_nl_noisecov_unbinned, noise_cov_preprocessed_recvbuf_cl_unbinned
            )
            np.save(
                manager.path_to_effectiv_bins_harmonic_compsep,
                nmt_bins.get_effective_ells()[bin_index_lminlmax],
            )

    if rank == root:
        logger.info("\n\nNoise covariance matrix computation step completed successfully.\n\n")


def main():
    parser = argparse.ArgumentParser(
        description="Pixel noise covariance estimater",
    )
    parser.add_argument("--config", type=Path, required=True, help="config file")

    args = parser.parse_args()
    config = Config.load_yaml(args.config)
    manager = DataManager(config)

    world, rank, size = get_world()
    if rank == 0:
        manager.dump_config()

    pixel_noisecov_estimation(manager, config)


if __name__ == "__main__":
    main()
