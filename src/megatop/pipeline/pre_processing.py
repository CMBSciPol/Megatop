import argparse
from megatop.metadata_manager import BBmeta, Timer
import IPython
import warnings
import healpy as hp
import numpy as np
from fgbuster.observation_helpers import (
    get_instrument,
    get_sky,
    get_observation,
    standardize_instrument,
)
import glob
import os
import sys
import matplotlib.pyplot as plt
import megatop.V3calc as V3
import copy
import time
import tracemalloc


def get_maps(args, id_split=None):
    # get path from maps and import them

    meta = BBmeta(args.globals)
    maps_list = meta.maps_list
    freq_maps = []
    for m in maps_list:
        path_m = meta.get_map_filename(m, id_split)
        map = hp.read_map(path_m, field=None).tolist()
        # MSS2 maps don't have the same nside from one frequency to another!!
        freq_maps.append(map)
    # freq_maps = np.array(freq_maps)
    freq_maps = np.array(freq_maps, dtype=object)

    return freq_maps


def read_maps(args):
    """
    To complete ...
    """
    meta = BBmeta(args.globals)
    cmb_fg_freq_maps_beamed = []
    for i, map in enumerate(meta.maps_list):
        fname = os.path.join(meta.map_directory, meta.map_sets[map]["file_root"] + ".fits")
        cmb_fg_freq_maps_beamed.append(hp.read_map(fname, field=None).tolist())
    return np.array(cmb_fg_freq_maps_beamed, dtype=object)


def read_maps_ben_sims(args, id_sim=0):
    """
    To complete ...
    """
    meta = BBmeta(args.globals)
    if hasattr(meta, "ben_unfiltered") and meta.ben_unfiltered:
        beamed_sky_unfiltered = np.load(
            meta.map_directory + "signal_unfiltered_freqs_nside128_" + str(id_sim).zfill(4) + ".npy"
        )
        # noise_maps = np.load(meta.map_directory + 'noise_nhits_freqs_nside128_'+str(id_sim).zfill(4)+'.npy')
        noise_maps = np.load(
            "/pscratch/sd/b/beringue/BB-AWG/MEGATOP/1224_sims_obsmat_freqs/noise_nhits_freqs_nside128_"
            + str(meta.id_sim).zfill(4)
            + ".npy"
        )
        cmb_fg_freq_maps_beamed = beamed_sky_unfiltered + noise_maps
    elif hasattr(meta, "noiseless_filtered") and meta.noiseless_filtered:
        print("NOISELESS FILTERED CASE")
        try:
            cmb_fg_freq_maps_beamed = np.load(
                meta.map_directory + "total_freqs_nside128_" + str(id_sim).zfill(4) + ".npy"
            )
        except FileNotFoundError:
            cmb_fg_freq_maps_beamed = np.load(
                meta.map_directory + "total_beamed_freqs_nside128_" + str(id_sim).zfill(4) + ".npy"
            )

        noise_maps = np.load(
            "/pscratch/sd/b/beringue/BB-AWG/MEGATOP/1224_sims_obsmat_freqs/noise_nhits_freqs_nside128_"
            + str(meta.id_sim).zfill(4)
            + ".npy"
        )
        cmb_fg_freq_maps_beamed = cmb_fg_freq_maps_beamed - noise_maps
    else:
        try:
            cmb_fg_freq_maps_beamed = np.load(
                meta.map_directory + "total_beamed_freqs_nside128_" + str(id_sim).zfill(4) + ".npy"
            )
        except FileNotFoundError:
            cmb_fg_freq_maps_beamed = np.load(
                meta.map_directory + "total_freqs_nside128_" + str(id_sim).zfill(4) + ".npy"
            )

    return cmb_fg_freq_maps_beamed


def CommonBeamConvAndNsideModification(args, freq_maps):
    """
    This function takes the frequency maps and applies the common beam correction, deconvolves the frequency beams,
    changes the NSIDE of the maps and includes the effect of the pixel window function.

    Args:
        args: The parser arguments, containing the path to the global parameters file to set up metadata manager.
        freq_maps (ndarray): The frequency maps, with shape (num_freq, num_stokes, num_pixels).

    Returns:
        freq_maps_out (ndarray): The frequency maps after the common beam correction,
                                 frequency beams decovolution, NSIDE change,
                                 and pixel window function effect,
                                 with shape (num_freq, num_stokes, num_pixels).
    """

    meta = BBmeta(args.globals)
    timer_beam = Timer()
    timer_beam.start("beam")
    map_dimensions = len(freq_maps.shape)
    freq_maps_out = []

    if args.verbose:
        print(
            "-> common beam correction and NSIDE change: correcting for frequency-dependent beams, convolving with a common beam, modifying NSIDE and include effect of pixel window function"
        )

    lmax_convolution = 3 * meta.general_pars["nside"]
    wpix_out = hp.pixwin(
        meta.general_pars["nside"], pol=True, lmax=lmax_convolution
    )  # Pixel window function of output maps
    Bl_gauss_common = hp.gauss_beam(
        np.radians(meta.pre_proc_pars["common_beam_correction"] / 60),
        lmax=lmax_convolution,
        pol=True,
    )

    for f in range(len(meta.frequencies)):
        # Input window function. WARNING: Must be done in the loop, as input map don't necessarily have the same nside (e.g. MSS2)
        wpix_in = hp.pixwin(
            hp.npix2nside(len(freq_maps[f, 0])), pol=True, lmax=lmax_convolution
        )  # Pixel window function of input maps
        wpix_in[1][0:2] = 1.0  # in order not to divide by 0

        # beam corrections
        Bl_gauss_fwhm = hp.gauss_beam(
            np.radians(meta.pre_proc_pars["fwhm"][f] / 60), lmax=lmax_convolution, pol=True
        )

        bl_correction = Bl_gauss_common / Bl_gauss_fwhm

        sm_corr_T = bl_correction[:, 0] * wpix_out[0] / wpix_in[0]
        sm_corr_P = bl_correction[:, 1] * wpix_out[1] / wpix_in[1]

        # map-->alm
        cmb_in_T = np.array(freq_maps[f, 0])
        cmb_in_Q = np.array(freq_maps[f, 1])
        cmb_in_U = np.array(freq_maps[f, 2])

        alm_in_T, alm_in_E, alm_in_B = hp.map2alm(
            [cmb_in_T, cmb_in_Q, cmb_in_U], lmax=lmax_convolution, pol=True, iter=10
        )
        # here lmax seems to play an important role

        # change beam and wpix
        alm_out_T = hp.almxfl(alm_in_T, sm_corr_T)
        alm_out_E = hp.almxfl(alm_in_E, sm_corr_P)
        alm_out_B = hp.almxfl(alm_in_B, sm_corr_P)

        # alm-->mapf
        cmb_out_T, cmb_out_Q, cmb_out_U = hp.alm2map(
            [alm_out_T, alm_out_E, alm_out_B],
            meta.general_pars["nside"],
            lmax=lmax_convolution,
            pixwin=False,
            fwhm=0.0,
            pol=True,
        )

        # a priori all the options are set to there default, even lmax which is computed wrt input alms
        out_map = np.array([cmb_out_T, cmb_out_Q, cmb_out_U])
        freq_maps_out.append(out_map)
    timer_beam.stop("beam", "Common beam convolution", args.verbose)
    return np.array(freq_maps_out)


def ApplyBinaryMask(args, freq_maps, use_UNSEEN=False):
    """
    This function applies the binary mask to the frequency maps.

    Args:
        args: The parser arguments, containing the path to the global parameters file to set up metadata manager.
        freq_maps (ndarray): The frequency maps to mask, with shape (num_freq, num_stokes, num_pixels).
        use_UNSEEN (bool): If True, the UNSEEN value is used for the masked pixels, otherwise the masked pixels are set to zero.

    Returns:
        freq_maps_masked (ndarray): The frequency maps after applying the binary mask, with shape (num_freq, num_stokes, num_pixels).
    """
    meta = BBmeta(args.globals)
    timer_mask = Timer()
    timer_mask.start("mask")
    binary_mask_path = meta.get_fname_mask("binary")
    binary_mask = hp.read_map(binary_mask_path, dtype=float)

    freq_maps_masked = copy.deepcopy(freq_maps)

    if use_UNSEEN:
        freq_maps_masked[:, np.where(binary_mask == 0)[0]] = hp.UNSEEN
    else:
        freq_maps_masked *= binary_mask
    timer_mask.stop("mask", "Masking", args.verbose)
    return freq_maps_masked


def save_preprocessed_maps(args, freq_maps_beamed_masked):
    """
    To complete ...
    """
    meta = BBmeta(args.globals)
    fname = os.path.join(meta.pre_process_directory, "freq_maps_preprocessed.npy")
    np.save(fname, freq_maps_beamed_masked)
    if args.verbose:
        print(f"Pre-processed maps (masked and beamed) saved to {fname}")


def check_preproc(args):
    """
    This function checks the pre-processed maps by comparing their Cls with the model Cls after pre-processing.
    It saves the different plots in the pre-processing subdirectory of the output plot directory.

    Args:
        args: The parser arguments, containing the path to the global parameters file to set up metadata manager.
        preproc_freq_maps (ndarray): The pre-processed frequency maps, with shape (num_freq, num_stokes, num_pixels).
        fg_freq_maps (ndarray): The foreground frequency maps, with shape (num_freq, num_stokes, num_pixels).
        fsky_binary (float): The fraction of sky covered by the binary mask.
        sim_num (int): The simulation number, used for the output file name.

    Returns:
        cl_preproc_freq_maps (ndarray): The pre-processed frequency maps Cls, with shape (num_freq, num_spectra [TT,EE,BB], num_ell).
        model_beamed_total (ndarray): The model beamed total Cls, with shape (num_freq, num_spectra [TT,EE,BB], num_ell).
    """
    meta = BBmeta(args.globals)
    print("NO CHECKS DONE FOR NOW !!!!!")  # TODO
    # Cl_cmb_model = get_Cl_CMB_model_from_meta(args)[:,:3] # Keeping only TT, EE, BB

    # cl_fg_freq_maps = []
    # cl_preproc_freq_maps = []

    # for f in range(len(meta.frequencies)):
    #     cl_fg_freq_maps.append( hp.anafast(fg_freq_maps[f]))
    #     cl_preproc_freq_maps.append( hp.anafast(preproc_freq_maps[f]))
    # cl_fg_freq_maps = np.array(cl_fg_freq_maps)[:,:3] # Keeping only TT, EE, BB
    # cl_preproc_freq_maps = np.array(cl_preproc_freq_maps)[:,:3] # Keeping only TT, EE, BB

    # lmax_convolution = 3*meta.general_pars['nside']
    # wpix_in = hp.pixwin( meta.general_pars['nside'],pol=True,lmax=lmax_convolution) # Pixel window function of input maps
    # wpix_out = hp.pixwin(meta.general_pars['nside'],pol=True,lmax=lmax_convolution) # Pixel window function of output maps
    # wpix_in[1][0:2] = 1. #in order not to divide by 0
    # Bl_gauss_common = hp.gauss_beam(np.radians(meta.pre_proc_pars['common_beam_correction']/60), lmax=lmax_convolution, pol=True)

    # beam_correction = []
    # for f in range(len(meta.general_pars['frequencies'])):
    #     #beam corrections
    #     Bl_gauss_fwhm = hp.gauss_beam( np.radians(meta.pre_proc_pars['fwhm'][f]/60), lmax=lmax_convolution, pol=True)

    #     bl_correction =  Bl_gauss_common / Bl_gauss_fwhm

    #     sm_corr_T = bl_correction[:,0] * wpix_out[0]/wpix_in[0]
    #     sm_corr_P = bl_correction[:,1] * wpix_out[1]/wpix_in[1]

    #     beam_correction.append([sm_corr_T, sm_corr_P, sm_corr_P])
    # beam_correction = np.array(beam_correction)

    # beam_correction_NODECONV_T = Bl_gauss_common[:,0] * wpix_out[0] #/wpix_in[0]
    # beam_correction_NODECONV_P = Bl_gauss_common[:,1] * wpix_out[1] #/wpix_in[1]
    # beam_correction_NODECONV = np.array([beam_correction_NODECONV_T, beam_correction_NODECONV_P, beam_correction_NODECONV_P])

    # # CMB spectra and fg maps (and their spectra) are not convolved by their respective frequency beams
    # # So to mimic and fit the pre-processed maps, only the common beams and pixel window functions must be applied.
    # CMB_fg_beamed_spectra = (Cl_cmb_model[...,:lmax_convolution] + cl_fg_freq_maps[...,:lmax_convolution]) * np.array([beam_correction_NODECONV[...,:lmax_convolution]**2])

    # # Noise after pre-processing model:
    # model_noise = get_Nl_white_noise(args, fsky_binary)
    # model_noise_beamed = model_noise[...,:lmax_convolution] * beam_correction[...,:lmax_convolution]**2

    # model_beamed_total = model_noise_beamed + CMB_fg_beamed_spectra

    # plotTTEEBB_diff(args, cl_preproc_freq_maps, model_beamed_total, os.path.join( meta.plot_dir_from_output_dir(meta.pre_process_directory_rel), 'preproc_check_SIM'+str(sim_num).zfill(5)+'.png'),
    #                 legend_labels=[r'Preproc $C_\ell$ from map $\nu=$', r'Model Cl after preproc $\nu=$'],
    #                 axis_labels=[r'$D_\ell \, [\mu K \, rad]^2$', r'$\frac{\Delta_{\ell}}{\rm{Input}_\ell}$'])

    # np.save(os.path.join(meta_sims.comb_spectra_directory, 'spectra_comb_freq_maps_SIM'+str(sim_num).zfill(5)+'.npy' ), cl_preproc_freq_maps)

    # return cl_preproc_freq_maps, model_beamed_total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="simplistic simulator")  # TODO change name ?
    parser.add_argument("--globals", type=str, help="Path to yaml with global parameters")
    parser.add_argument(
        "--use_mpi",
        action="store_true",
        help="Use MPI instead of for loops to pre-process multiple maps, or simulate multiple sims.",
    )
    parser.add_argument("--plots", action="store_true", help="Plot the generated maps if True.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    meta = BBmeta(args.globals)

    if hasattr(meta, "ben_sims") and meta.ben_sims:
        print("Reading ben sims")
        input_maps = read_maps_ben_sims(args, id_sim=meta.id_sim)
    else:
        input_maps = read_maps(args)
    if args.verbose:
        print("input_maps shape = ", input_maps.shape)

    if np.all(
        np.array(meta.pre_proc_pars["common_beam_correction"])
        == np.array(meta.pre_proc_pars["fwhm"])
    ):
        print("Common beam correction is the same as the input beam, no need to apply it.")
        print("WARNING: this is mostly for testing it might not actually represent the real noise")
        convolved_maps = input_maps.astype("float64")
    else:
        convolved_maps = CommonBeamConvAndNsideModification(args, input_maps)
    masked_convolved_maps = ApplyBinaryMask(args, convolved_maps)
    check_preproc(args)
    save_preprocessed_maps(args, masked_convolved_maps)

    print("\n\nPre-processing step completed succesfully\n\n")

    # # MPI VARIABLES
    # mpi = args.use_mpi

    # if mpi:
    #     try:
    #         from mpi4py import MPI
    #         comm=MPI.COMM_WORLD
    #         size=comm.Get_size()
    #         rank=comm.rank
    #         barrier=comm.barrier
    #         root=0
    #         mpi = True
    #         if args.verbose: print("MPI TRUE, SIZE = ", size,", RANK = ", rank,'\n')

    #     except (ModuleNotFoundError, ImportError) as e:
    #         # Error handling
    #         print('ERROR IN MPI:', e)
    #         print('Proceeding without MPI\n')
    #         mpi = False
    #         rank=0
    #         pass

    # tracemalloc.start()
    # if args.verbose: print('Memory usage at the start is: ', tracemalloc.get_traced_memory())

    # if args.sims:
    #     if args.verbose: print('Simulating maps ...')
    #     meta_sims = BBmeta(args.sims)

    #     if not mpi:
    #         freq_maps_sim_list = []
    #         for sim_num in range(meta_sims.general_pars['nsims']):

    #             if args.verbose: print('simulating maps sim number: ', sim_num + 1, '/', meta_sims.general_pars['nsims'], ' (for loop, NOT MPI)')
    #             if args.verbose: print('Memory usage before simulation is: ', tracemalloc.get_traced_memory())

    #             freq_maps, noise_maps, fg_freq_maps, cmb_sky, CMB_fg_freq_maps_beamed, fsky_binary = MakeSims(args)

    #             if args.verbose: print('Memory usage after simulation is: ', tracemalloc.get_traced_memory())

    #             freq_maps_sim_list.append(freq_maps)

    #             if args.verbose: print('saving map sims ...')
    #             np.save(os.path.join(meta_sims.comb_directory, 'comb_freq_maps_SIM'+str(sim_num).zfill(5)+'.npy' ),
    #                      freq_maps )
    #             np.save(os.path.join(meta_sims.cmb_directory, 'cmb_maps_SIM'+str(sim_num).zfill(5)+'.npy' ),
    #                     cmb_sky )
    #             np.save(os.path.join(meta_sims.cmb_beamed_directory, 'cmb_beamed_freq_maps_SIM'+str(sim_num).zfill(5)+'.npy' ),
    #                     CMB_fg_freq_maps_beamed )
    #             np.save(os.path.join(meta_sims.fg_directory, 'fg_freq_maps_SIM'+str(sim_num).zfill(5)+'.npy' ),
    #                     fg_freq_maps )
    #             np.save(os.path.join(meta_sims.noise_directory, 'noise_freq_maps_SIM'+str(sim_num).zfill(5)+'.npy' ),
    #                     noise_maps )

    #     if mpi:
    #         if meta_sims.general_pars['nsims'] != size:
    #             exit('ERROR: nsims must be equal to size in MPI mode. nsims = '+ str(meta_sims.general_pars['nsims'])+'  size = '+ str(size))

    #         if args.verbose: print('simulating maps sim number: ', rank + 1, '/', meta_sims.general_pars['nsims'], ' (MPI)')
    #         if args.verbose: print('Memory usage before simulation is: ', tracemalloc.get_traced_memory())
    #         freq_maps, noise_maps, fg_freq_maps, cmb_sky, CMB_fg_freq_maps_beamed, fsky_binary = MakeSims(args)
    #         if args.verbose: print('Memory usage after simulation is: ', tracemalloc.get_traced_memory())
    #         np.save(os.path.join(meta_sims.comb_directory, 'comb_freq_maps_SIM'+str(rank).zfill(5)+'.npy' ),
    #                 freq_maps )
    #         np.save(os.path.join(meta_sims.cmb_directory, 'cmb_maps_SIM'+str(rank).zfill(5)+'.npy' ),
    #                 cmb_sky )
    #         np.save(os.path.join(meta_sims.cmb_beamed_directory, 'cmb_beamed_freq_maps_SIM'+str(rank).zfill(5)+'.npy' ),
    #                  CMB_fg_freq_maps_beamed )
    #         np.save(os.path.join(meta_sims.fg_directory, 'fg_freq_maps_SIM'+str(rank).zfill(5)+'.npy' ),
    #                 fg_freq_maps )
    #         np.save(os.path.join(meta_sims.noise_directory, 'noise_freq_maps_SIM'+str(rank).zfill(5)+'.npy' ),
    #                 noise_maps )

    # else:
    #     if args.verbose: print('Importing maps ...')
    #     # TODO: MPI
    #     # TODO: formating for importing real maps
    #     freq_maps = get_maps(args)

    # if args.sims and args.plots:
    #     if args.verbose:print('checking sims...')
    #     # This only checks the last simulations
    #     # TODO: in MPI do all cases or just one?
    #     if args.verbose: print('Memory usage before checking sims is: ', tracemalloc.get_traced_memory())
    #     if mpi and rank==0:
    #         check_sims(args, cmb_sky, noise_maps, freq_maps, fg_freq_maps, CMB_fg_freq_maps_beamed, fsky_binary)
    #     elif not mpi:
    #         check_sims(args, cmb_sky, noise_maps, freq_maps, fg_freq_maps, CMB_fg_freq_maps_beamed, fsky_binary)
    #     if args.verbose: print('Memory usage after checking sims is: ', tracemalloc.get_traced_memory())

    # if not mpi:
    #     for sim_num in range(meta_sims.general_pars['nsims']):
    #         if args.verbose: print('Memory usage before pre-processing is: ', tracemalloc.get_traced_memory())
    #         if args.verbose: print('Pre-precessing freq-maps #', sim_num + 1, ' out of ', meta_sims.general_pars['nsims'])
    #         freq_maps_common_beamed = CommonBeamConvAndNsideModification(args, freq_maps_sim_list[sim_num])
    #         freq_maps_common_beamed_masked = ApplyBinaryMask(args, freq_maps_common_beamed)
    #         if args.verbose: print('Memory usage after pre-processing is: ', tracemalloc.get_traced_memory())
    #         if args.verbose: print('saving pre-processed maps ...')
    #         np.save(os.path.join(meta.pre_process_directory, 'freq_maps_common_beamed_masked'+str(sim_num).zfill(5)+'.npy' ),
    #                 freq_maps_common_beamed_masked )
    #         np.save(os.path.join(meta.pre_process_directory, 'freq_maps_common_beamed'+str(sim_num).zfill(5)+'.npy' ),
    #                 freq_maps_common_beamed )

    #     if args.sims and args.plots:
    #         if args.verbose: print('checking pre-processed maps...\n')
    #         # This only checks the last simulations
    #         cl_preproc_freq_maps, model_beamed_total = check_preproc( args, freq_maps_common_beamed, fg_freq_maps, fsky_binary, sim_num=sim_num)

    # if mpi:
    #     # TODO: check mpi version

    #     if args.verbose: print('Pre-precessing freq-maps #', rank + 1, ' out of ', meta_sims.general_pars['nsims'], ' (MPI)')
    #     if args.verbose: print('Memory usage before pre-processing is: ', tracemalloc.get_traced_memory())
    #     freq_maps_common_beamed = CommonBeamConvAndNsideModification(args, freq_maps)
    #     freq_maps_common_beamed_masked = ApplyBinaryMask(args, freq_maps_common_beamed)
    #     if args.verbose: print('Memory usage after pre-processing is: ', tracemalloc.get_traced_memory())

    #     if args.verbose: print('saving pre-processed maps ...\n')
    #     np.save(os.path.join(meta.pre_process_directory, 'freq_maps_common_beamed_masked'+str(rank).zfill(5)+'.npy' ),
    #             freq_maps_common_beamed_masked )

    #     if not meta_sims.noise_sim_pars['include_nhits']:
    #         # If nhits is used for the noise, all maps should be masked.
    #         np.save(os.path.join(meta.pre_process_directory, 'freq_maps_common_beamed'+str(rank).zfill(5)+'.npy' ),
    #                 freq_maps_common_beamed )

    #     if args.sims and args.plots:
    #         if args.verbose: print('checking pre-processed maps...\n')
    #         if args.verbose: print('Memory usage before checking pre-processed maps is: ', tracemalloc.get_traced_memory())
    #         cl_preproc_freq_maps, model_beamed_total = check_preproc( args, freq_maps_common_beamed, fg_freq_maps, fsky_binary, sim_num=rank)
    #         if args.verbose: print('Memory usage after checking pre-processed maps is: ', tracemalloc.get_traced_memory())
    #         # Ensure recvbuf is contiguous
    #         # to make sure the comm.Gather() works correctly

    #         cl_preproc_freq_maps = np.ascontiguousarray(cl_preproc_freq_maps)

    #         recvbuf = None
    #         if rank == 0:
    #             shape_recvbuf = (size,) + cl_preproc_freq_maps.shape
    #             recvbuf = np.empty(shape_recvbuf)

    #             # Ensure recvbuf is contiguous
    #             # to make sure the comm.Gather() works correctly
    #             recvbuf = np.ascontiguousarray(recvbuf)

    #         comm.Gather(cl_preproc_freq_maps, recvbuf, root=0)

    #         if rank == 0:
    #             if args.verbose: print('checking MEAN preproc results...\n')

    #             mean_cl_preproc_freq_maps = np.mean(recvbuf, axis=0)
    #             plotTTEEBB_diff(args, mean_cl_preproc_freq_maps, model_beamed_total,
    #                             os.path.join( meta.plot_dir_from_output_dir(meta.pre_process_directory_rel), 'mean_preproc_cl_check.png'),
    #                             legend_labels=[r'Mean preproc $C_\ell$ from map $\nu=$', r'Model Cl after preproc $\nu=$'],
    #                             axis_labels=[r'$D_\ell \, [\mu K \, rad]^2$', r'$\frac{\Delta_{\ell}}{\rm{Input}_\ell}$'])

    # if args.verbose: print('Memory usage at the end is: ', tracemalloc.get_traced_memory())
    # if rank == 0:
    #     print('\n\nPre-Processing step completed succesfully\n\n')
