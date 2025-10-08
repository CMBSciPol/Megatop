import argparse
import tracemalloc
from pathlib import Path

import healpy as hp
import numpy as np
from scipy.linalg import sqrtm

from megatop import Config, DataManager
from megatop.utils import Timer, logger
from megatop.utils.binning import load_nmt_binning
from megatop.utils.mpi import MPISUM, get_world
from megatop.utils.preproc import alm_common_beam, common_beam_and_nside
from megatop.utils.spectra import initialize_nmt_workspace, spectra_from_namaster
from megatop.utils.utils import MemoryUsage


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
    test_alm_TF_noise = True
    if test_alm_TF_noise:
        noise_cov_alm_preprocessed = np.zeros(
            [
                len(config.frequencies),
                2,
                hp.Alm.getsize(config.parametric_sep_pars.harmonic_lmax - 1),
            ]
        )

    if config.parametric_sep_pars.use_harmonic_compsep:
        nmt_bins = load_nmt_binning(manager)
        bin_index_lminlmax = np.load(manager.path_to_binning, allow_pickle=True)[
            "bin_index_lminlmax"
        ]

        ell_min_namaster = config.parametric_sep_pars.harmonic_lmin
        ell_max_namaster = config.parametric_sep_pars.harmonic_lmax

        mask_analysis = hp.read_map(manager.path_to_analysis_mask)

        # if config.masks_pars.DEBUG_output_apod_binary_mask:
        #     logger.warning(
        #         "DEBUG: Using apodized binary mask for harmonic component separation (PIXEL NOISE COV step) "
        #     )
        #     mask_analysis = hp.read_map(manager.path_to_apod_binary_mask)
        logger.warning("Normalizing analysis mask to 1, TODO: remove after merge")
        # TODO: remove after merge
        mask_analysis /= np.max(mask_analysis)  # normalize the mask to 1

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

        if (
            np.all(np.array(config.pre_proc_pars.common_beam_correction) == np.array(config.beams))
            or config.pre_proc_pars.DEBUGskippreproc
        ):
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
            if config.parametric_sep_pars.harmonic_delta_ell != 1:
                # use_beam = True
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
                    return_all_spectra=config.pre_proc_pars.correct_for_TF,
                )

                if config.pre_proc_pars.correct_for_TF:
                    logger.warning("DEBUG: Including transfer function in the pre-processed alms. ")

                    output_noise_spectra = np.zeros(
                        [len(config.frequencies), 3, nmt_bins.get_n_bands()]
                    )
                    output_noise_spectra_unbined = np.zeros(
                        [len(config.frequencies), 3, noise_spectra_unbined.shape[-1]]
                    )

                    reduced_TF_from_preproc = np.load(
                        manager.get_path_to_preprocessed_reduced_TF(),
                        allow_pickle=True,
                    )
                    for f, tf_path in enumerate(manager.get_TF_filenames()):
                        if tf_path == Path():
                            logger.warning(
                                f"DEBUG: Transfer function for frequency p{config.frequencies[f]} is not provided, skipping."
                            )
                            output_noise_spectra[f, 0] = noise_spectra[f, 0] * 0
                            output_noise_spectra[f, 1] = noise_spectra[f, 0]
                            output_noise_spectra[f, 2] = noise_spectra[f, 3]
                            output_noise_spectra_unbined[f, 0] = noise_spectra_unbined[f, 0] * 0
                            output_noise_spectra_unbined[f, 1] = noise_spectra_unbined[f, 0]
                            output_noise_spectra_unbined[f, 2] = noise_spectra_unbined[f, 3]
                            continue

                        reduced_TF = True
                        # TODO: remove reduced_TF option, or if needed, make it a parameter in config
                        if reduced_TF:
                            # Using the same limited elements as for the preproc step
                            # Since preproc uses alms and not spectra we only have alm_E, and alm_B
                            """
                            inv_tf_ = np.zeros_like(inv_tf)
                            if config.pre_proc_pars.sum_TF_column:
                                inv_tf_sum = np.sum(
                                    inv_tf, axis=1
                                )  # summing over column to get all the contribution xy-->ab (EE-->EE + EB-->EE + BE-->EE + BB-->EE etc)
                                inv_tf_[:, 0, 0] = inv_tf_sum[:, 0]  # EE->EE
                                inv_tf_[:, 1, 1] = inv_tf_sum[:, 1]  # EB->EB
                                inv_tf_[:, 2, 2] = inv_tf_sum[:, 2]  # BE->BE
                                inv_tf_[:, 3, 3] = inv_tf_sum[:, 3]  # BB->BB
                            else:
                                inv_tf_[:, 0, 0] = inv_tf[:, 0, 0]
                                inv_tf_[:, 1, 1] = inv_tf[:, 1, 1]
                                # inv_tf_[:, 1, 1] = inv_tf[:, 0, -1]
                                inv_tf_[:, 2, 2] = inv_tf[:, 2, 2]
                                # inv_tf_[:, 2, 2] = inv_tf[:, -1, 0]
                                inv_tf_[:, 3, 3] = inv_tf[:, 3, 3]
                            inv_tf = inv_tf_

                            inv_tf_reduced = get_reduced_TF_for_Cl(None, transfer)
                            inv_tf = inv_tf_reduced
                            """
                            inv_tf = get_reduced_TF_for_Cl(
                                reduced_TF_from_preproc["inv_sqrt_tf_bin_freq"][f], None
                            )
                        else:
                            logger.info(f"Loading transfer function from {tf_path}")
                            transfer = np.load(tf_path, allow_pickle=True)["full_tf"]

                            inv_tf = np.linalg.inv([T_ell.T for T_ell in transfer.T])[
                                :, -4:, -4:
                            ]  # taking only polarised components
                            # careful with the transpose here, transfer is not symetric

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

            if test_alm_TF_noise:
                analysis_mask = hp.read_map(manager.path_to_analysis_mask)
                logger.warning("Normalizing analysis mask to 1, TODO: remove after merge")
                # TODO: remove after merge
                analysis_mask /= np.max(analysis_mask)

                freq_beams = config.beams
                common_beam = config.pre_proc_pars.common_beam_correction
                if config.pre_proc_pars.DEBUGskippreproc:
                    freq_beams = np.array([0.0] * len(config.frequencies))
                    common_beam = 0.0

                freq_alms_convolved = alm_common_beam(
                    nside=config.nside,
                    common_beam=common_beam,
                    frequency_beams=freq_beams,
                    freq_maps=np.array(noise_freq_maps),
                    analysis_mask=analysis_mask,
                    harmonic_analysis_lmax=config.parametric_sep_pars.harmonic_lmax,
                )

                freq_alms_convolved_TF_corrected = np.einsum(
                    "fijl,fjl->fil",
                    reduced_TF_from_preproc["inv_sqrt_tf_lm_freq"],
                    freq_alms_convolved,
                )

                noise_cov_alm_preprocessed += np.real(
                    freq_alms_convolved_TF_corrected * np.conj(freq_alms_convolved_TF_corrected)
                )

                manager.get_path_to_preprocessed_noise_alms(sub=id_real).parent.mkdir(
                    exist_ok=True, parents=True
                )
                logger.info("Saving pre-processed noise alms to disk")
                np.save(
                    manager.get_path_to_preprocessed_noise_alms(sub=id_real),
                    freq_alms_convolved_TF_corrected,
                )

    if comm is not None:
        noise_cov_preprocessed_recvbuf = MPISUM(noise_cov_preprocessed, comm, rank, root)
        if config.parametric_sep_pars.use_harmonic_compsep:
            noise_cov_preprocessed_recvbuf_cl = MPISUM(cl_noise_cov_preprocessed, comm, rank, root)
            noise_cov_preprocessed_recvbuf_cl_unbinned = MPISUM(
                cl_noise_cov_preprocessed_unbinned, comm, rank, root
            )
            if test_alm_TF_noise:
                noise_cov_preprocessed_recvbuf_alm = MPISUM(
                    noise_cov_alm_preprocessed, comm, rank, root
                )
    else:
        noise_cov_preprocessed_recvbuf = noise_cov_preprocessed
        if config.parametric_sep_pars.use_harmonic_compsep:
            noise_cov_preprocessed_recvbuf_cl = cl_noise_cov_preprocessed
            noise_cov_preprocessed_recvbuf_cl_unbinned = cl_noise_cov_preprocessed_unbinned
            if test_alm_TF_noise:
                noise_cov_preprocessed_recvbuf_alm = noise_cov_alm_preprocessed

    if rank == root:
        # Average noise_cov and noise_cov_preprocessed over nsims
        noise_cov_preprocessed_mean = noise_cov_preprocessed_recvbuf / int_n_sim
        if config.parametric_sep_pars.use_harmonic_compsep:
            noise_cov_preprocessed_mean_cl = noise_cov_preprocessed_recvbuf_cl / int_n_sim
            noise_cov_preprocessed_recvbuf_cl_unbinned = (
                noise_cov_preprocessed_recvbuf_cl_unbinned / int_n_sim
            )
            if test_alm_TF_noise:
                noise_cov_preprocessed_mean_alm = noise_cov_preprocessed_recvbuf_alm / int_n_sim

    else:
        noise_cov_preprocessed_mean = None
        if config.parametric_sep_pars.use_harmonic_compsep:
            noise_cov_preprocessed_mean_cl = None
            noise_cov_preprocessed_recvbuf_cl_unbinned = None
            if test_alm_TF_noise:
                noise_cov_preprocessed_mean_alm = None

    if rank == root:
        manager.path_to_covar.mkdir(exist_ok=True, parents=True)
        np.save(manager.path_to_pixel_noisecov, noise_cov_preprocessed_mean)
        if config.parametric_sep_pars.use_harmonic_compsep:
            np.save(manager.path_to_nl_noisecov, noise_cov_preprocessed_mean_cl)
            np.save(
                manager.path_to_nl_noisecov_unbinned, noise_cov_preprocessed_recvbuf_cl_unbinned
            )
            if test_alm_TF_noise:
                np.save(manager.path_to_noisecov_alm, noise_cov_preprocessed_mean_alm)

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
