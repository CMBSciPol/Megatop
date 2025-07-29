import argparse
import tracemalloc
from pathlib import Path

import healpy as hp
import numpy as np

from megatop import Config, DataManager
from megatop.utils import Timer, logger
from megatop.utils.binning import load_nmt_binning
from megatop.utils.mpi import MPISUM, get_world
from megatop.utils.preproc import common_beam_and_nside
from megatop.utils.spectra import initialize_nmt_workspace, spectra_from_namaster
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

    if config.parametric_sep_pars.use_harmonic_compsep:
        # Initilizing the binning sccheme used in the harmonic component separation in namaster
        # bin_low, bin_high, bin_centre = create_binning(
        #     config.nside,
        #     config.parametric_sep_pars.harmonic_delta_ell,
        #     end_first_bin=config.parametric_sep_pars.harmonic_delta_ell,
        # )
        # nmt_bins = nmt.NmtBin.from_edges(bin_low, bin_high + 1)
        nmt_bins = load_nmt_binning(manager)
        bin_index_lminlmax = np.load(manager.path_to_binning, allow_pickle=True)[
            "bin_index_lminlmax"
        ]

        ell_min_namaster = config.parametric_sep_pars.harmonic_lmin
        ell_max_namaster = config.parametric_sep_pars.harmonic_lmax
        # bin_index_lminlmax = np.where(
        #     (bin_low >= ell_min_namaster) & (bin_high <= ell_max_namaster)
        # )[0]

        # Bins from Carlos BBMASTER paper:
        # USE_BBMASTER_BINS = False

        # if config.parametric_sep_pars.DEBUGuse_BBMASTER_bin:
        #     logger.warning("Using EXTERNAL BBMASTER bins for the harmonic component separation.")

        #     nmt_bins = nmt.NmtBin.from_nside_linear(config.nside, nlb=10, is_Dell=False)
        #     bin_index_lminlmax = np.where(
        #         (nmt_bins.get_effective_ells() >= ell_min_namaster)
        #         & (nmt_bins.get_effective_ells() <= ell_max_namaster)
        #     )[0]

        mask_analysis = hp.read_map(manager.path_to_analysis_mask)

        if config.masks_pars.DEBUG_output_apod_binary_mask:
            logger.warning(
                "DEBUG: Using apodized binary mask for harmonic component separation (PIXEL NOISE COV step) "
            )
            mask_analysis = hp.read_map(manager.path_to_apod_binary_mask)
        logger.warning("Normalizing analysis mask to 1, TODO: remove after merge")
        # TODO: remove after merge
        mask_analysis /= np.max(mask_analysis)  # normalize the mask to 1
        # if config.parametric_sep_pars.DEBUGnorm_mask:
        #     mask_analysis /= np.max(mask_analysis)  # normalize the mask to 1

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

        # DEBUGtruncatealms = True,

        if (
            np.all(np.array(config.pre_proc_pars.common_beam_correction) == np.array(config.beams))
            or config.pre_proc_pars.DEBUGskippreproc
        ):  # and not DEBUGtruncatealms:
            logger.info(
                "Common beam correction is the same as the input beam, no need to apply it."
            )
            logger.info(
                "WARNING: this is mostly for testing it might not actually represent the real noise"
            )
            noise_freq_maps_preprocessed = np.array(noise_freq_maps)

        else:
            noise_freq_maps = np.array(noise_freq_maps, dtype=object)
            # DEBUGlm_range = [
            #     config.parametric_sep_pars.harmonic_lmin,
            #     config.parametric_sep_pars.harmonic_lmax,
            # ]

            noise_freq_maps_preprocessed = common_beam_and_nside(
                nside=config.nside,
                common_beam=config.pre_proc_pars.common_beam_correction,
                frequency_beams=config.beams,
                freq_maps=noise_freq_maps,
                # DEBUGtruncatealms=DEBUGtruncatealms,
                # DEBUGlm_range=DEBUGlm_range,
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
            if config.parametric_sep_pars.harmonic_delta_ell != 1:
                # use_beam = True
                if config.parametric_sep_pars.DEBUGnamaster_deconv:
                    # import IPython; IPython.embed()
                    common_beam = hp.gauss_beam(
                        np.radians(config.pre_proc_pars.common_beam_correction / 60.0),
                        lmax=3 * config.nside,
                        pol=True,
                    )[
                        :-1, 1
                    ]  # taking only the GRAD/ELECTRIC/E polarization beam (it is equal to the  CURL/MAGNETIC/B polarization beam)
                    # beam4namaster = np.tile(common_beam, (len(config.frequencies), 1))
                    # beam4namaster = np.tile(beam4namaster, (len(config.frequencies), 1))

                    beam4namaster = np.array(
                        [
                            hp.gauss_beam(np.radians(beam / 60), lmax=3 * config.nside, pol=True)[
                                :-1, 1
                            ]
                            / common_beam
                            for beam in config.beams
                        ]
                    )
                    workspaceff = None
                    input_namaster_noise_maps = np.array(
                        noise_freq_maps
                    )  # TODO: there is some redundancy in the case where all beam = 0 and common beam = 0

                    if config.pre_proc_pars.DEBUGskippreproc:
                        beam4namaster = None
                    # test_maps = np.tile(noise_freq_maps_preprocessed[0], (6,1,1))

                    # correct_TF = False
                    # if correct_TF:
                    #     BBTF = np.load('/lustre/work/jost/SO_MEGATOP/harmonic_test_Nl_std_beam_nhits_obsmatBBMASER_namaster/TF_FirstDayEveryMonth_Full_nside512_fpthin8_pwf_beam.npz', allow_pickle=True)
                    #     transfer = BBTF['tf']
                    #     nside_native = 512
                    #     nmt_bins_native = nmt.NmtBin.from_nside_linear(nside_native, nlb=10, is_Dell=False)

                    #     transfer_flat = transfer.reshape(-1, transfer.shape[-1])
                    #     unbin_transfer_flat = nmt_bins_native.unbin_cell(transfer_flat)
                    #     unbin_transfer = unbin_transfer_flat.reshape(transfer.shape[0], transfer.shape[1], -1)[...,:config.parametric_sep_pars.harmonic_lmax]

                    #     inv_unbined_TF = np.zeros_like(unbin_transfer)
                    #     # Ignoring the first two bins which are always 0
                    #     # Keeping them to 0, they will be ignored in the rest of the code anyways
                    #     inv_unbined_TF[...,2:] = np.linalg.inv(unbin_transfer[...,2:].T).T

                else:
                    beam4namaster = None
                    input_namaster_noise_maps = noise_freq_maps_preprocessed

                noise_spectra, noise_spectra_unbined = spectra_from_namaster(
                    input_namaster_noise_maps,
                    mask_analysis,
                    workspaceff,
                    nmt_bins,
                    compute_cross_freq=False,
                    purify_e=False,
                    purify_b=False,
                    beam=beam4namaster,
                    return_all_spectra=config.pre_proc_pars.DEBUGinclude_TF,
                )

                if config.pre_proc_pars.DEBUGinclude_TF:
                    logger.warning("DEBUG: Including transfer function in the pre-processed alms. ")

                    # nside_native = 512

                    # nmt_bins_native = nmt.NmtBin.from_nside_linear(
                    #     nside_native, nlb=10, is_Dell=False
                    # )
                    # Checking if bins from noise_spectra computation and from TF computation match
                    # if not np.all(
                    #     nmt_bins_native.get_effective_ells()[
                    #         nmt_bins_native.get_effective_ells() < nmt_bins.lmax
                    #     ]
                    #     == nmt_bins.get_effective_ells()
                    # ):
                    #     Error_msg = "Binning scheme from noise_spectra computation and from TF computation do not match. "
                    #     raise Exception(Error_msg)

                    # common_bins = nmt_bins_native.get_effective_ells() < nmt_bins.lmax
                    output_noise_spectra = np.zeros(
                        [len(config.frequencies), 3, nmt_bins.get_n_bands()]
                    )  # sum(common_bins)
                    output_noise_spectra_unbined = np.zeros(
                        [len(config.frequencies), 3, noise_spectra_unbined.shape[-1]]
                    )

                    # import IPython; IPython.embed()

                    for f, tf_path in enumerate(manager.get_TF_filenames()):
                        if tf_path == Path():
                            logger.warning(
                                f"DEBUG: Transfer function for frequency {config.frequencies[f]} is not provided, skipping."
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

                        inv_tf = np.linalg.inv([T_ell.T for T_ell in transfer.T])[
                            :, -4:, -4:
                        ]  # taking only polarised compoenents
                        # [
                        #     common_bins
                        # ]  # careful with the transpose here, transfer is not symetric

                        noise_spectra_TF_corrected = np.einsum(
                            "lij,jl->il",
                            inv_tf,
                            noise_spectra[f],
                        )

                        noise_spectra_TF_corrected_unbined = nmt_bins.unbin_cell(
                            noise_spectra_TF_corrected
                        )
                        output_noise_spectra[f, 0] = noise_spectra_TF_corrected[0] * 0
                        output_noise_spectra[f, 1] = noise_spectra_TF_corrected[0]
                        output_noise_spectra[f, 2] = noise_spectra_TF_corrected[3]

                        output_noise_spectra_unbined[f, 0] = (
                            noise_spectra_TF_corrected_unbined[0] * 0
                        )
                        output_noise_spectra_unbined[f, 1] = noise_spectra_TF_corrected_unbined[0]
                        output_noise_spectra_unbined[f, 2] = noise_spectra_TF_corrected_unbined[3]
                noise_spectra = output_noise_spectra
                noise_spectra_unbined = output_noise_spectra_unbined

            else:
                logger.warning(
                    "Using harmonic delta ell = 1, this is not recommended for noise spectra estimation. Healpy is used in this case."
                )
                noise_spectra = np.array(
                    [
                        hp.anafast(noise_freq_maps_preprocessed[i])[:3]
                        for i in range(len(config.frequencies))
                    ]
                )
                noise_spectra_unbined = noise_spectra.copy()
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
