import argparse
# from megatop import BBmeta, utils
from megatop import BBmeta


import numpy as np
import os
import healpy as hp
import matplotlib.pyplot as plt
import glob
import IPython
from matplotlib import cm
import math
import megatop.V3calc as V3
from pre_processing import CommonBeamConvAndNsideModification, plotTTEEBB_diff, get_Nl_white_noise
from tqdm import tqdm
import time

# import sys
# sys.path.append('/global/cfs/cdirs/sobs/users/krach/BBSims/NOISE_20201207/')
# from combine_noise import compute_noise_factors, combine_noise_maps

def plot_cov_matrix(args, noise_cov_mean, file_name, mask_unseen=None, norm=None):
    meta = BBmeta(args.globals)

    plot_dir = meta.plot_dir_from_output_dir(meta.covmat_directory_rel)
    cmap = cm.RdBu
    cmap.set_under("w")

    cols = math.ceil(math.sqrt(noise_cov_mean.shape[0])) # making a grid to display all the frequency maps
    rows = math.ceil(noise_cov_mean.shape[0] / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(16, 9))

    for ax, f in zip(axes.flatten(), range(noise_cov_mean.shape[0])):
        plt.sca(ax) #setting current axe
        noise_cov_mean_ = noise_cov_mean[f].copy()
        if mask_unseen is not None:
            noise_cov_mean_[np.where(mask_unseen==0)[0]] = hp.UNSEEN
        hp.mollview(noise_cov_mean_, cmap=cmap, cbar=True, hold=True, 
                    title=r'Noise cov map $\nu={}$ GHz'.format(meta.frequencies[f]),
                    norm=norm) 
        hp.graticule()

    map_noise_cov_save_path = os.path.join(plot_dir, file_name)
    plt.savefig(map_noise_cov_save_path)
    plt.close()

def plot_hist_freqmaps(args,freq_maps, save_name, plot_gauss=False, bins=100, binary_mask=None):
    meta = BBmeta(args.globals)

    if binary_mask is None:
        binary_mask = np.ones(freq_maps.shape[-1], dtype=bool)

    fig, ax = plt.subplots(1,3,figsize=(16,9))
    for f in range(len(meta.frequencies)):
        ax[0].hist(freq_maps[f,0,binary_mask], bins=bins, histtype='step', density=True)
        ax[1].hist(freq_maps[f,1,binary_mask], bins=bins, histtype='step', density=True)
        ax[2].hist(freq_maps[f,2,binary_mask], bins=bins, histtype='step', label=r'$\nu = $'+str(meta.frequencies[f])+' GHz', density=True)
    if plot_gauss:
        x = np.linspace(-5,5,1000)
    
        ax[0].plot(x, 1/np.sqrt(2*np.pi)*np.exp(-x**2/2))
        ax[1].plot(x, 1/np.sqrt(2*np.pi)*np.exp(-x**2/2))
        ax[2].plot(x, 1/np.sqrt(2*np.pi)*np.exp(-x**2/2), label=r'$\mathcal{N}(0,1) $')

    plt.legend()

    ax[0].set_title('T')
    ax[1].set_title('Q')
    ax[2].set_title('U')

    plot_dir = meta.plot_dir_from_output_dir(meta.covmat_directory_rel)
    plt.savefig( os.path.join(plot_dir, save_name))
    plt.close()

def noise_covariance_estimation(meta, map_shape, instrument, nhits):
    """
    Estimation of the noise covariance matrix
    """
    noise_cov = np.zeros(map_shape)
    noise_cov_beamed = np.zeros(map_shape)
    nhits_nz = np.where(nhits!=0)[0]

    for i_sim in range(meta.config['Nsims_bias']):

        if meta.config['external_noise_sims']!='' or meta.config['Nico_noise_combination']:
            noise_maps = np.zeros(map_shape)
            print('NOISE COV ESTIMATION LOADING EXTERNAL NOISE-ONLY MAPS, SIM#',i_sim,'/',meta.config['Nsims_bias'])

            if meta.config['Nico_noise_combination']:
                if meta.config['knee_mode'] == 2 : knee_mode_loc = None
                else: knee_mode_loc = meta.config['knee_mode']
                factors = compute_noise_factors(meta.config['sensitivity_mode'], knee_mode_loc)

            for f in range(len(instrument.frequency)):
                print('loading noise map for frequency ', str(int(instrument.frequency[f])))

                if meta.config['Nico_noise_combination']:
                    noise_loc = combine_noise_maps(i_sim, instrument.frequency[f], factors)
                else:
                    noise_loc = hp.read_map(glob.glob(os.path.join(meta.config['external_noise_sims'],'SO_SAT_'+str(int(instrument.frequency[f]))+'_noise_FULL_*_white_20201207.fits'))[0], field=None)

                alms = hp.map2alm(noise_loc, lmax=3*meta.config['nside'])
                Bl_gauss_pix = hp.gauss_beam( hp.nside2resol(meta.config['nside']), lmax=2*meta.config['nside'])        
                for alm_ in alms: hp.almxfl(alm_, Bl_gauss_pix, inplace=True)             
                noise_maps[3*f:3*(f+1),:] = hp.alm2map(alms, meta.config['nside'])  

                if ((not meta.config['no_inh']) and (meta.config['Nico_noise_combination'])):
                    # renormalize the noise map to take into account the effect of inhomogeneous noise
                    noise_maps[3*f:3*(f+1),nhits_nz] /= np.sqrt(nhits[nhits_nz]/np.max(nhits[nhits_nz]))

        elif meta.config['noise_option']=='white_noise':
            np.random.seed(i_sim)
            nlev_map = np.zeros(map_shape)
            for f in range(len(instrument.frequency)):
                nlev_map[3*f:3*f+3,:] = np.array([instrument.depth_i[f], instrument.depth_p[f], instrument.depth_p[f]])[:,np.newaxis]*np.ones((3,map_shape[-1]))
            nlev_map /= hp.nside2resol(meta.config['nside'], arcmin=True)
            noise_maps = np.random.normal(freq_maps*0.0, nlev_map, map_shape)

        elif meta.config['noise_option']=='no_noise': 
            pass

        noise_maps_beamed = noise_maps*1.0

        if meta.config['common_beam_correction']!=0.0:

            Bl_gauss_common = hp.gauss_beam( np.radians(meta.config['common_beam_correction']/60), lmax=2*meta.config['nside'])        
            for f in range(len(instrument.frequency)):
                Bl_gauss_fwhm = hp.gauss_beam( np.radians(instrument.fwhm[f]/60), lmax=2*meta.config['nside'])

                alms_n = hp.map2alm(noise_maps_beamed[3*f:3*(f+1),:], lmax=3*meta.config['nside'])
                for alms_ in alms_n:
                    hp.almxfl(alms_, Bl_gauss_common/Bl_gauss_fwhm, inplace=True)             
                noise_maps_beamed[3*f:3*(f+1),:] = hp.alm2map(alms_n, meta.config['nside'])   

        noise_cov += noise_maps**2
        noise_cov_beamed += noise_maps_beamed**2

    return noise_cov/meta.config['Nsims_bias'], noise_cov_beamed/meta.config['Nsims_bias']




def GetNoiseCov(args, apply_pixwin=False):
    # MPI VARIABLES
    mpi = args.use_mpi
    
    if mpi:
        try:
            from mpi4py import MPI
            comm=MPI.COMM_WORLD
            size=comm.Get_size()
            rank=comm.rank
            barrier=comm.barrier
            root=0
            mpi = True
        except (ModuleNotFoundError, ImportError) as e:
            # Error handling
            print('ERROR IN MPI:', e)
            print('Proceeding without MPI\n')
            mpi = False
            rank=0
            pass
    else:
        rank=0
        root=0
    meta = BBmeta(args.globals)


    if args.sims:
        print('sims=True: Computing noise covariance from OnTheFlySims')
        meta_sims = BBmeta(args.sims)
        nsims = meta_sims.general_pars['nsims']

        noise_cov = np.zeros( [ len(meta.general_pars['frequencies']), 3,hp.nside2npix(meta_sims.map_sim_pars['nside_sim'])])

        noise_cov_preprocessed = np.zeros([ len(meta.general_pars['frequencies']), 3,hp.nside2npix(meta.general_pars['nside'])])

        if args.plots:
            freq_noise_maps_pre_processed_array = []
            freq_noise_maps_array = []

        if not mpi:
            nsims_iter = nsims
        else:
            nsims_iter = 1 #TODO: implement version with nsims_iter = nsims//size so size doesn't have to be equal to nsims.
            if nsims != size:
                raise('ERROR: nsims must be equal to size in MPI mode')

        for sim_num in range(nsims_iter):
            if mpi:
                sim_num = rank
            print('NOISE COV ESTIMATION LOADING EXTERNAL NOISE-ONLY MAPS, SIM#',sim_num+1,'/',nsims)
            freq_noise_maps = np.load(os.path.join(meta_sims.output_dirs['root'], meta_sims.output_dirs['noise_directory'], 'noise_freq_maps_SIM'+str(sim_num).zfill(5)+'.npy' ) )
            if apply_pixwin:
                for f in range(freq_noise_maps.shape[0]):
                    alms_T, alms_E, alms_B = hp.map2alm(freq_noise_maps[f], lmax=3*meta_sims.map_sim_pars['nside_sim'])
                    Bl_gauss_pix = hp.gauss_beam( hp.nside2resol(meta_sims.map_sim_pars['nside_sim']), lmax=3*meta_sims.map_sim_pars['nside_sim'], pol=True)     

                    alms_T_pixwin = hp.almxfl(alms_T, Bl_gauss_pix[0], inplace=False)             
                    alms_E_pixwin = hp.almxfl(alms_E, Bl_gauss_pix[1], inplace=False)             
                    alms_B_pixwin = hp.almxfl(alms_B, Bl_gauss_pix[1], inplace=False)             
                    freq_noise_maps[f] = hp.alm2map([alms_T_pixwin, alms_E_pixwin, alms_B_pixwin],
                                                    nside= meta_sims.map_sim_pars['nside_sim'],lmax=3*meta_sims.map_sim_pars['nside_sim'],
                                                    pixwin=False,fwhm=0.0,pol=True)  

            mask_path_before_preproc = os.path.join(meta_sims.mask_directory, meta.masks["binary_mask"])
            mask_beforepreproc = hp.read_map(mask_path_before_preproc)
            # freq_noise_maps  *= mask_beforepreproc
            # freq_noise_maps[...,np.where(mask_beforepreproc==0)[0]] = hp.UNSEEN
            # freq_noise_maps_maskedUNSEEN_pre_processed = CommonBeamConvAndNsideModification(args, freq_noise_maps_masked_unseen)
            # plot_cov_matrix(args, freq_noise_maps_maskedUNSEEN_pre_processed[:,1],
            #                 'freq_noise_maps_masked_unseen_pre_processed_Q.png', mask_unseen=mask)
            # plot_cov_matrix(args, freq_noise_maps_maskedUNSEEN_pre_processed[:,1]-freq_noise_maps_pre_processed[:,1],
            #                 'freq_noise_maps_DIFF_with_masked_unseen_pre_processed_Q.png', mask_unseen=mask)            

            
            if meta.noise_cov_pars['include_nhits']:
                print('')
                print('Applying nhits to both freq noise maps')


                nhits_map = hp.read_map(meta_sims._get_nhits_map_name())
                nhits_map_rescaled = nhits_map / max(nhits_map)
                freq_noise_maps /= np.sqrt(nhits_map_rescaled)
                freq_noise_maps[...,np.where(nhits_map_rescaled==0)[0]] = 0 # hp.UNSEEN
                
                threshold_centre_patch = 0.8
                binary_mask_centre_nhits_sims = np.zeros(nhits_map_rescaled.shape, dtype=bool)
                binary_mask_centre_nhits_sims[np.where(nhits_map_rescaled >= threshold_centre_patch)[0]] = True


                binary_mask_from_nhits_nside_sims = np.ones(nhits_map.shape, dtype=bool)
                binary_mask_from_nhits_nside_sims[np.where(nhits_map==0)[0]] = 0
                fsky_from_binary_mask_from_nhits_nside_sims = sum(binary_mask_from_nhits_nside_sims) / len(binary_mask_from_nhits_nside_sims)

            else:
                binary_mask_from_nhits_nside_sims = None
                binary_mask_centre_nhits_sims = None
                fsky_from_binary_mask_from_nhits_nside_sims = 1

            
            noise_cov += freq_noise_maps**2


            freq_noise_maps_pre_processed = CommonBeamConvAndNsideModification(args, freq_noise_maps)
            print('CommonBeamConvAndNsideModification DONE Sim#',sim_num+1,'/',nsims)

            if args.plots:
                freq_noise_maps_pre_processed_array.append(freq_noise_maps_pre_processed)
                freq_noise_maps_array.append(freq_noise_maps)

            noise_cov_preprocessed += freq_noise_maps_pre_processed**2
        
        if mpi:
            if rank==root:
                noise_cov_recvbuf = np.zeros_like(noise_cov)
                noise_cov_preprocessed_recvbuf = np.zeros_like(noise_cov_preprocessed)
            else:
                noise_cov_recvbuf = None
                noise_cov_preprocessed_recvbuf = None
            
            noise_cov = np.ascontiguousarray(noise_cov) 
            noise_cov_preprocessed = np.ascontiguousarray(noise_cov_preprocessed) 

            start_reduce_1 = time.time()
            print('MPI REDUCING NOISE_COV...')
            comm.Reduce(noise_cov, noise_cov_recvbuf, op=MPI.SUM, root=root)
            print('time reduce noise_cov = ', time.time()-start_reduce_1)
            
            start_reduce_2 = time.time()
            print('MPI REDUCING NOISE_COV_PREPROCESSED...')
            comm.Reduce(noise_cov_preprocessed, noise_cov_preprocessed_recvbuf, op=MPI.SUM, root=root)
            print('Time reduce noise_cov_preprocessed = ', time.time()-start_reduce_2)

            if rank==root:
                print('MPI REDUCING DONE (in root)...')
                noise_cov_mean = noise_cov_recvbuf/nsims
                noise_cov_preprocessed_mean = noise_cov_preprocessed_recvbuf/nsims
                # return noise_cov_mean, noise_cov_preprocessed_mean
            else:
                print('MPI RETURNING NONE FOR NON ROOT PROCESSES...')
                # return None, None # only root returns the noise_cov and noise_cov_preprocessed, not great syntax though, might lead to problems TODO: fix this
                noise_cov_mean = None
                noise_cov_preprocessed_mean = None
        else:
            noise_cov_mean = noise_cov/nsims
            noise_cov_preprocessed_mean = noise_cov_preprocessed/nsims


        binary_mask_from_nhits_preproc = None
        if meta.noise_cov_pars['include_nhits'] and rank==root:
            print('')
            print('Converting nhits (version with nisde from analysis) to binary mask and apply to noise_cov_preprocessed_mean')

            nhits_map_after_preproc = hp.read_map(meta._get_nhits_map_name())
            nhits_map_after_preproc_rescaled = nhits_map_after_preproc / max(nhits_map_after_preproc)
            
            binary_mask_centre_nhits_preproc = np.zeros(nhits_map_after_preproc_rescaled.shape, dtype=bool)
            binary_mask_centre_nhits_preproc[np.where(nhits_map_after_preproc_rescaled >= threshold_centre_patch)[0]] = True


            binary_mask_from_nhits_preproc = np.ones(nhits_map_after_preproc.shape, dtype=bool)
            binary_mask_from_nhits_preproc[np.where(nhits_map_after_preproc==0)[0]] = 0
            fsky_from_binary_mask_from_nhits_preproc = sum(binary_mask_from_nhits_preproc) / len(binary_mask_from_nhits_preproc)


            noise_cov_preprocessed_mean *= binary_mask_from_nhits_preproc
        else:
            binary_mask_from_nhits_preproc = None
            binary_mask_centre_nhits_preproc = None
            fsky_from_binary_mask_from_nhits_preproc = 1

        # Gatheting freq_noise_maps_pre_processed_array and freq_noise_maps_array to root, WARNING: might use a lot of memory for large nsims
        if mpi:
            print('MPI Gathering freq_noise_maps_pre_processed_array and freq_noise_maps_array to root, rank=',rank)
            freq_noise_maps_pre_processed_array = np.ascontiguousarray(freq_noise_maps_pre_processed_array[0]) 
            freq_noise_maps_array = np.ascontiguousarray(freq_noise_maps_array[0]) 

            recvbuf_freq_noise_maps_pre_processed_array = None
            recvbuf_freq_noise_maps_array = None
            if rank == 0:
                shape_recvbuf_freq_noise_maps_pre_processed_array = (size,) + freq_noise_maps_pre_processed_array.shape
                shape_recvbuf_freq_noise_maps_array = (size,) + freq_noise_maps_array.shape
                print('shape_recvbuf_freq_noise_maps_pre_processed_array = ', shape_recvbuf_freq_noise_maps_pre_processed_array)
                print('shape_recvbuf_freq_noise_maps_array = ', shape_recvbuf_freq_noise_maps_array)
                recvbuf_freq_noise_maps_pre_processed_array = np.empty(shape_recvbuf_freq_noise_maps_pre_processed_array)
                recvbuf_freq_noise_maps_array = np.empty(shape_recvbuf_freq_noise_maps_array)

                # Ensure recvbuf is contiguous
                # to make sure the comm.Gather() works correctly
                recvbuf_freq_noise_maps_pre_processed_array = np.ascontiguousarray(recvbuf_freq_noise_maps_pre_processed_array)
                recvbuf_freq_noise_maps_array = np.ascontiguousarray(recvbuf_freq_noise_maps_array)

            comm.Gather(freq_noise_maps_pre_processed_array, recvbuf_freq_noise_maps_pre_processed_array, root=0)
            comm.Gather(freq_noise_maps_array, recvbuf_freq_noise_maps_array, root=0)

            # redefining the name of the arrays:
            freq_noise_maps_pre_processed_array = recvbuf_freq_noise_maps_pre_processed_array
            freq_noise_maps_array = recvbuf_freq_noise_maps_array
            print('MPI Gathering freq_noise_maps_pre_processed_array and freq_noise_maps_array to root DONE, rank=',rank)


        if rank==root:
            print('\n===================================')
            print('shape freq_noise_maps_pre_processed_array = ', np.shape(freq_noise_maps_pre_processed_array))
            print('shape freq_noise_maps_array = ', np.shape(freq_noise_maps_array))
            print('shape noise_cov_mean = ', np.shape(noise_cov_mean))
            print('shape noise_cov_preprocessed_mean = ', np.shape(noise_cov_preprocessed_mean))
            print('===================================\n')



        if args.plots and rank==root:
            print('PLOTTING...')

            noise_cov_preprocessed_mean_pixwin = np.empty(noise_cov_preprocessed_mean.shape)
            for f in range(len(meta.general_pars['frequencies'])):
                alms_T, alms_E, alms_B = hp.map2alm(noise_cov_preprocessed_mean[f], lmax=3*meta.nside, pol=True, iter=10)
                # Bl_gauss_pix = hp.gauss_beam( hp.nside2resol(meta.nside), lmax=3*meta.nside, pol=True)     
                wpix_in = hp.pixwin( meta_sims.general_pars['nside'], pol=True, lmax=3*meta.nside) # Pixel window function of input maps

                alms_T_pixwin = hp.almxfl(alms_T, wpix_in[0], inplace=False)             
                alms_E_pixwin = hp.almxfl(alms_E, wpix_in[1], inplace=False)             
                alms_B_pixwin = hp.almxfl(alms_B, wpix_in[1], inplace=False)             
                noise_cov_preprocessed_mean_pixwin[f] = hp.alm2map([alms_T_pixwin, alms_E_pixwin, alms_B_pixwin],
                                                nside= meta.nside,lmax=3*meta.nside,
                                                pixwin=False,fwhm=0.0,pol=True)  

        
            freq_noise_maps_pre_processed_array = np.array(freq_noise_maps_pre_processed_array)
            freq_noise_maps_array = np.array(freq_noise_maps_array)

            mask_path = os.path.join(meta.mask_directory, meta.masks["binary_mask"])
            mask = hp.read_map(mask_path)
            fsky_mask = sum(mask) / len(mask) # only works if binary mask...



            std_uK_arcmin_noise_cov_mean = CheckWhiteNoiselvl(args, noise_cov_mean, fsky_mask, mask=binary_mask_centre_nhits_sims)            

            std_map = np.sqrt(noise_cov_mean)

            noise_sim_from_cov = np.random.normal(0*std_map, std_map, noise_cov_mean.shape)

            
            ratio_noisecov_sim_from_cov, cl_ratio_noisecov_sim_from_cov = CheckNoiseSpectra(args, noise_cov_mean, noise_sim_from_cov, mask=binary_mask_from_nhits_nside_sims)

            print('std of ratio, should be close to 1 = ', np.std(ratio_noisecov_sim_from_cov[...,binary_mask_centre_nhits_sims], axis=-1))
            # ratio_noisecov_sim, cl_ratio_noisecov_sim = CheckNoiseSpectra(args, noise_cov_mean, freq_noise_maps_array[i])
            plot_allspectra(args, cl_ratio_noisecov_sim_from_cov/hp.nside2resol(meta_sims.nside)**2 / fsky_from_binary_mask_from_nhits_nside_sims, 'ratio_noisecov_map_SIM_FROM_COV.png', 
                            legend_labels=[r'data $C_\ell$ $\nu=$', r'model $C_\ell$ $\nu=$'], 
                            axis_labels=[r'$C_\ell \, [\mu K \, rad]^2$', r'$C_\ell \, [\mu K \, rad]^2$'])


            plot_hist_freqmaps(args, ratio_noisecov_sim_from_cov, 'hist_ratio_noisecov_map_SIM_FROM_COV.png', plot_gauss=True, binary_mask=binary_mask_centre_nhits_sims)

            Noise_ell = get_Nl_white_noise(args, fsky_binary=fsky_mask)



            # ell_V3, N_ell_P_SA, Map_white_noise_levels = V3.so_V3_SA_noise(
            #             sensitivity_mode = meta_sims.noise_sim_pars['sensitivity_mode'],
            #             one_over_f_mode = 2, # fixed to None since we only use white noise here
            #             SAC_yrs_LF = meta_sims.noise_sim_pars['SAC_yrs_LF'], f_sky = fsky_mask, 
            #             ell_max = meta.general_pars['lmax'], delta_ell=1,
            #             beam_corrected=False, remove_kluge=False, CMBS4='' )

            # N_ell_P_SA_reshape = np.ones([N_ell_P_SA.shape[0],3,N_ell_P_SA.shape[-1]])
            # N_ell_P_SA_reshape *=  N_ell_P_SA[:,np.newaxis,:]






            cl_noise_cov_FREQ_mean = []
            cl_sqrt_noise_cov_FREQ_mean = []
            cl_noise_sim_from_cov_FREQ = []
            for f in tqdm(range(len(meta.general_pars['frequencies']))):
                sqrt_cov = np.sqrt(noise_cov_mean[f])
                sqrt_cov[np.isnan(sqrt_cov)] = 0

                cl_sqrt_noise_cov_FREQ_mean.append( hp.anafast(sqrt_cov, lmax = 3*meta.general_pars['nside'], pol=True))
                cl_noise_cov_FREQ_mean.append( hp.anafast(noise_cov_mean[f], lmax = 3*meta.general_pars['nside'], pol=True))
                cl_noise_sim_from_cov_FREQ.append( hp.anafast(noise_sim_from_cov[f], lmax = 3*meta.general_pars['nside'], pol=True))

            cl_noise_cov_FREQ_mean = np.array(cl_noise_cov_FREQ_mean)
            cl_sqrt_noise_cov_FREQ_mean = np.array(cl_sqrt_noise_cov_FREQ_mean)
            cl_noise_sim_from_cov_FREQ = np.array(cl_noise_sim_from_cov_FREQ)

            plotTTEEBB_diff(args,
                            cl_noise_sim_from_cov_FREQ,
                            Noise_ell[...,:97],
                            os.path.join(meta.plot_dir_from_output_dir(meta.covmat_directory_rel),
                                         'noise_SIMFromCov_vs_model_cl.png'), 
                            legend_labels=[r'$C_\ell$ $\nu=$', r'Noise model $C_\ell$ $\nu=$'], 
                            axis_labels=[r'$C_\ell \, [\mu K \, rad]^2$', r'$\frac{\Delta_{\ell}}{\rm{Input}_\ell}$'],
                            use_D_ell=False)

            plotTTEEBB_diff(args,
                            #  Noise_ell[...,:93],
                            cl_noise_cov_FREQ_mean,# * (fsky_mask*4*np.pi)**2,
                            # N_ell_P_SA_reshape, 
                            Noise_ell[...,:97],   
                            os.path.join(meta.plot_dir_from_output_dir(meta.covmat_directory_rel),
                                         'noise_cov_mean_vs_model_cl.png'), 
                            legend_labels=[r'$D_\ell$ $\nu=$', r'cov $D_\ell$ $\nu=$'], 
                            axis_labels=[r'$D_\ell \, [\mu K \, rad]^2$', r'$\frac{\Delta_{\ell}}{\rm{Input}_\ell}$'],
                            use_D_ell=False)

            
            # Plotting noise sim / sqrt(noise cov) spectra BEFORE pre-processing
            nsims_plot = 2 # number of sims to plot 
            if nsims_plot>nsims:
                nsims_plot = nsims
            for i in tqdm(range(nsims_plot)):
                ratio_noisecov_sim, cl_ratio_noisecov_sim = CheckNoiseSpectra(args, noise_cov_mean, freq_noise_maps_array[i], mask=binary_mask_from_nhits_nside_sims)
                print('Std of ratio before pre-processing SIM#'+str(i)+', should be close to 1 = \n', np.std(ratio_noisecov_sim[...,binary_mask_from_nhits_nside_sims], axis=-1))

                plot_hist_freqmaps(args, ratio_noisecov_sim, 'hist_ratio_noisecov_map_SIM'+str(i)+'.png', plot_gauss=True, binary_mask=binary_mask_from_nhits_nside_sims)

                # ratio_noisecov_sim, cl_ratio_noisecov_sim = CheckNoiseSpectra(args, noise_cov_mean, freq_noise_maps_array[i])
                plot_allspectra(args, cl_ratio_noisecov_sim/hp.nside2resol(meta_sims.nside)**2 / fsky_from_binary_mask_from_nhits_nside_sims, 'ratio_noisecov_map_SIM'+str(i)+'.png', 
                                legend_labels=[r'label data $C_\ell$ $\nu=$', r'label model $C_\ell$ $\nu=$'], 
                                axis_labels=[r'$C_\ell \, [\mu K \, rad]^2$', r'$C_\ell \, [\mu K \, rad]^2$'])

            ratio_noisecov_sim_testMeanPixel, cl_ratio_noisecov_sim_testMeanPixel = CheckNoiseSpectra(args, np.mean(noise_cov_mean[...,binary_mask_from_nhits_nside_sims],axis=-1)[...,np.newaxis]*np.ones(noise_cov_mean.shape), 
                                                                                                      freq_noise_maps_array[i], 
                                                                                                      mask=binary_mask_from_nhits_nside_sims)
            print('Std of ratio before pre-processing (MEAN NOISE COV) SIM#'+str(i)+', should be close to 1 = \n', np.std(ratio_noisecov_sim_testMeanPixel[...,binary_mask_from_nhits_nside_sims], axis=-1))

            plot_hist_freqmaps(args, ratio_noisecov_sim_testMeanPixel, 'hist_ratio_MEANnoisecov_map_SIM'+str(i)+'.png', plot_gauss=True, binary_mask=binary_mask_from_nhits_nside_sims)

            # Plotting noise sim / sqrt(noise cov) spectra AFTER pre-processing

            lmax_convolution = 3*meta.general_pars['nside']
            wpix_in = hp.pixwin( meta_sims.nside ,pol=True, lmax=lmax_convolution) # Pixel window function of input maps
            wpix_out = hp.pixwin(meta.nside, pol=True, lmax=lmax_convolution) # Pixel window function of output maps
            wpix_in[1][0:2] = 1. #in order not to divide by 0
            Bl_gauss_common = hp.gauss_beam(np.radians(meta.pre_proc_pars['common_beam_correction']/60), lmax=lmax_convolution, pol=True)
            
            effective_beam_freq = []
            for f in range(len(meta.frequencies)):
                #beam corrections
                Bl_gauss_fwhm = hp.gauss_beam( np.radians(meta.pre_proc_pars['fwhm'][f]/60), lmax=lmax_convolution, pol=True)

                bl_correction =  Bl_gauss_common / Bl_gauss_fwhm

                sm_corr_T = bl_correction[:,0] * wpix_out[0]/wpix_in[0]
                sm_corr_P = bl_correction[:,1] * wpix_out[1]/wpix_in[1]
                effective_beam_freq.append([sm_corr_T, sm_corr_P, sm_corr_P, 
                                            np.sqrt(sm_corr_T*sm_corr_P), np.sqrt(sm_corr_P*sm_corr_P) ,np.sqrt(sm_corr_T*sm_corr_P)])
            effective_beam_freq = np.array(effective_beam_freq)
            # plot_allspectra(args, effective_beam_freq**2,
            #                 'Effective_beam_test.png', 
            #                 legend_labels=[r'Beam $C_\ell$ $\nu=$'],
            #                 axis_labels=[r'$C_\ell \, [\mu K \, rad]^2$', r'$C_\ell \, [\mu K \, rad]^2$'])

            common_beam_and_pixwin = np.array([Bl_gauss_common[:,0] * wpix_out[0], 
                                               Bl_gauss_common[:,1] * wpix_out[1], 
                                               Bl_gauss_common[:,1] * wpix_out[1],
                                               Bl_gauss_common[:,3] * np.sqrt(wpix_out[0]*wpix_out[1]),
                                               Bl_gauss_common[:,0] * wpix_out[1],
                                               Bl_gauss_common[:,3] * np.sqrt(wpix_out[0]*wpix_out[1])])
            common_beam_and_pixwin_FREQ = np.empty(effective_beam_freq.shape)
            common_beam_and_pixwin_FREQ[:] = common_beam_and_pixwin
            
            # plot_allspectra(args, common_beam_and_pixwin_FREQ**2,
            #                 'Common_beam_and_pixwin_test.png', 
            #                 legend_labels=[r'Beam $C_\ell$ $\nu=$'],
            #                 axis_labels=[r'$C_\ell \, [\mu K \, rad]^2$', r'$C_\ell \, [\mu K \, rad]^2$'])  

            common_beam = np.array([Bl_gauss_common[:,0], Bl_gauss_common[:,1],Bl_gauss_common[:,2],
                                            Bl_gauss_common[:,3],Bl_gauss_common[:,1],Bl_gauss_common[:,3]])
            common_beam_FREQ = np.empty(effective_beam_freq.shape)
            common_beam_FREQ[:] = common_beam_and_pixwin
            # plot_allspectra(args, common_beam_FREQ,
            #                 'Common_beam_test.png', 
            #                 legend_labels=[r'Beam $C_\ell$ $\nu=$'],
            #                 axis_labels=[r'$C_\ell \, [\mu K \, rad]^2$', r'$C_\ell \, [\mu K \, rad]^2$'])    

            cl_ratio_noisecov_preprocessed_sim_array = []
            mean_beam_offset = np.mean(effective_beam_freq, axis=-1)


            # ============ This is for testing but is seems useless, to be removed =================
            use_fsky_from_binary_mask_from_nhits_preproc_for_noise_computation = False
            if use_fsky_from_binary_mask_from_nhits_preproc_for_noise_computation:
                fsky_test_lvl = fsky_from_binary_mask_from_nhits_preproc
            else:
                fsky_test_lvl = fsky_mask
            
            test_lvl = CheckWhiteNoiselvl(args, noise_cov_preprocessed_mean, fsky_test_lvl, mask=binary_mask_centre_nhits_preproc)
            test_lvl = np.array([test_lvl[:,0], test_lvl[:,1], test_lvl[:,1], np.sqrt(test_lvl[:,0]*test_lvl[:,1]), np.sqrt(test_lvl[:,1]*test_lvl[:,1]), np.sqrt(test_lvl[:,0]*test_lvl[:,1])])
            #  ========================================================================================
            for i in tqdm(range(nsims_plot)):
                ratio_noisecov_sim_preprocessed, cl_ratio_noisecov_preprocessed_sim = CheckNoiseSpectra(args, noise_cov_preprocessed_mean, freq_noise_maps_pre_processed_array[i], mask=binary_mask_from_nhits_preproc)

                print('Std of ratio after pre-processing SIM#'+str(i)+', should be close to 1 = \n', np.std(ratio_noisecov_sim_preprocessed[...,binary_mask_from_nhits_preproc], axis=-1))
                mean_beam_offset_ell = np.ones(cl_ratio_noisecov_preprocessed_sim.shape)* mean_beam_offset[:, :, np.newaxis]
                
                plot_hist_freqmaps(args, ratio_noisecov_sim_preprocessed, 'hist_ratio_noisecov_preprocessed_map_SIM'+str(i)+'.png', plot_gauss=True, binary_mask=binary_mask_from_nhits_preproc)

                plot_allspectra(args, 
                                cl_ratio_noisecov_preprocessed_sim/hp.nside2resol(meta.nside)**2 / effective_beam_freq**2 / fsky_from_binary_mask_from_nhits_preproc, #* mean_beam_offset_ell**4,
                                 'ratio_noisecov_preprocessed_map_SIM'+str(i)+'.png', 
                                legend_labels=[r'label data $C_\ell$ $\nu=$', r'label model $C_\ell$ $\nu=$'], 
                                axis_labels=[r'$C_\ell \, [\mu K \, rad]^2$', r'$C_\ell \, [\mu K \, rad]^2$'])
                cl_ratio_noisecov_preprocessed_sim_array.append( cl_ratio_noisecov_preprocessed_sim)
            cl_ratio_noisecov_preprocessed_sim_array = np.array(cl_ratio_noisecov_preprocessed_sim_array)

            '''
            IPython.embed()
            
            # Plotting pre-processed noise map sims
            cl_noise_map_preprocessed_array = []
            for i in tqdm(range(2)):
                cl_noise_map_preprocessed = []
                for f in range(len(meta.general_pars['frequencies'])):
                    cl_noise_map_preprocessed.append( hp.anafast(freq_noise_maps_pre_processed_array[i][f], pol=True, lmax = 3*meta.nside))
                plot_allspectra(args, np.array(cl_noise_map_preprocessed), 'noise_Cl_map_preprocessed_SIM'+str(i)+'.png', 
                                legend_labels=[r'label data $C_\ell$ $\nu=$', r'label model $C_\ell$ $\nu=$'], 
                                axis_labels=[r'$C_\ell \, [\mu K \, rad]^2$', r'$C_\ell \, [\mu K \, rad]^2$']) 
                cl_noise_map_preprocessed_array.append( cl_noise_map_preprocessed)
            cl_noise_map_preprocessed_array = np.array(cl_noise_map_preprocessed_array)

            # Plotting noise map sims BEFORE pre-processing
            cl_noise_map_array = []
            for i in tqdm(range(2)):
                cl_noise_map = []
                for f in range(len(meta.general_pars['frequencies'])):
                    cl_noise_map.append( hp.anafast(freq_noise_maps_array[i][f], pol=True, lmax = 3*meta.nside))
                plot_allspectra(args, np.array(cl_noise_map), 'noise_Cl_map_SIM'+str(i)+'.png', 
                                legend_labels=[r'label data $C_\ell$ $\nu=$', r'label model $C_\ell$ $\nu=$'], 
                                axis_labels=[r'$C_\ell \, [\mu K \, rad]^2$', r'$C_\ell \, [\mu K \, rad]^2$'], plot_hline_one=False) 
                cl_noise_map_array.append( cl_noise_map)
            cl_noise_map_array = np.array(cl_noise_map_array)

            
            plotTTEEBB_diff(args,
                            cl_noise_map_array[0],
                            Noise_ell[...,:3*meta.nside+1],
                            os.path.join(meta.plot_dir_from_output_dir(meta.covmat_directory_rel),
                                         'noise_sim_cl_vs_nl_model.png'), 
                            legend_labels=[r'$Noise Sim C_\ell$ $\nu=$', r'Model Noise $N_\ell$ $\nu=$'], 
                            axis_labels=[r'$C_\ell \, [\mu K \, rad]^2$', r'$\frac{\Delta_{\ell}}{\rm{Input}_\ell}$'],
                            use_D_ell=False)

            #  Computing sqrt(noise covariance) spectra 
            cl_noise_cov_FREQ_preprocessed_mean = []
            for f in tqdm(range(len(meta.general_pars['frequencies']))):
                # sqrt_cov_preprocessed = np.sqrt(noise_cov_preprocessed_mean[f])
                # sqrt_cov_preprocessed[np.isnan(sqrt_cov_preprocessed)] = 0
                # cl_noise_cov_FREQ_preprocessed_mean.append( hp.anafast(sqrt_cov_preprocessed, lmax = 3*meta.general_pars['nside'], pol=True))
                cl_noise_cov_FREQ_preprocessed_mean.append( hp.anafast(noise_cov_preprocessed_mean[f], lmax = 3*meta.general_pars['nside'], pol=True))

            cl_noise_cov_FREQ_preprocessed_mean = np.array(cl_noise_cov_FREQ_preprocessed_mean)



            # Plotting comparison of noise cov and noise map sims BEFORE pre-processing
            plotTTEEBB_diff(args, cl_noise_map_array[0], cl_noise_cov_FREQ_mean, \
                            os.path.join(meta.plot_dir_from_output_dir(meta.covmat_directory_rel),
                                         'noise_cov_mean_cl_diff.png'), 
                legend_labels=[r'data $D_\ell$ $\nu=$', r'cov $D_\ell$ $\nu=$'], 
                axis_labels=[r'$D_\ell \, [\mu K \, rad]^2$', r'$\frac{\Delta_{\ell}}{\rm{Input}_\ell}$'])         

            plotTTEEBB_diff(args, cl_noise_cov_FREQ_mean, Noise_ell[...,:97]**2 / (4*np.pi), 
                            os.path.join(meta.plot_dir_from_output_dir(meta.covmat_directory_rel),
                            'noise_cov_mean_vs_Model.png'), 
                legend_labels=[r'Noise cov $C_\ell$ $\nu=$', r'Model cov $C_\ell$ $\nu=$'], 
                axis_labels=[r'$C_\ell \, [\mu K \, rad]^2$', r'$\frac{\Delta_{\ell}}{\rm{Input}_\ell}$'],
                use_D_ell=False)  

            # Plotting comparison of noise cov and noise map sims AFTER pre-processing
            plotTTEEBB_diff(args, cl_noise_map_preprocessed_array[0], cl_noise_cov_FREQ_preprocessed_mean, 
                            os.path.join(meta.plot_dir_from_output_dir(meta.covmat_directory_rel),
                                         'noise_cov_preprocessed_mean_cl_diff.png'), 
                legend_labels=[r'data $D_\ell$ $\nu=$', r'cov $D_\ell$ $\nu=$'], 
                axis_labels=[r'$D_\ell \, [\mu K \, rad]^2$', r'$\frac{\Delta_{\ell}}{\rm{Input}_\ell}$'])   

            # cl_noise_cov_FREQ_preprocessed_mean_pixwin = []
            # for f in tqdm(range(len(meta.general_pars['frequencies']))):
            #     sqrt_cov_preprocessed_pixwin = np.sqrt(noise_cov_preprocessed_mean_pixwin[f])
            #     sqrt_cov_preprocessed_pixwin[np.isnan(sqrt_cov_preprocessed_pixwin)] = 0
            #     cl_noise_cov_FREQ_preprocessed_mean_pixwin.append( hp.anafast(sqrt_cov_preprocessed_pixwin, lmax = 3*meta.general_pars['nside'], pol=True))
            # cl_noise_cov_FREQ_preprocessed_mean_pixwin = np.array(cl_noise_cov_FREQ_preprocessed_mean_pixwin)


            plotTTEEBB_diff(args, cl_noise_map_array[0], cl_noise_cov_FREQ_preprocessed_mean_pixwin, '/global/u2/j/jost/Megatop/pipeline/noise_cov_preprocessed_mean_pixwin_cl_diff.png', 
                            legend_labels=[r'label data $D_\ell$ $\nu=$', r'label cov $D_\ell$ $\nu=$'], 
                            axis_labels=[r'$D_\ell \, [\mu K \, rad]^2$', r'$\frac{\Delta_{\ell}}{\rm{Input}_\ell}$'])

            plotTTEEBB_diff(args, cl_noise_map_array[0], cl_noise_cov_FREQ_preprocessed_mean, '/global/u2/j/jost/Megatop/pipeline/noise_cov_preprocessed_mean_cl_diff.png', 
                            legend_labels=[r'label data $D_\ell$ $\nu=$', r'label cov $D_\ell$ $\nu=$'], 
                            axis_labels=[r'$D_\ell \, [\mu K \, rad]^2$', r'$\frac{\Delta_{\ell}}{\rm{Input}_\ell}$'])

            plotTTEEBB_diff(args, cl_noise_map_array[0], cl_noise_cov_FREQ_mean, '/global/u2/j/jost/Megatop/pipeline/noise_cov_mean_cl_diff.png', 
                            legend_labels=[r'label data $D_\ell$ $\nu=$', r'label cov $D_\ell$ $\nu=$'], 
                            axis_labels=[r'$D_\ell \, [\mu K \, rad]^2$', r'$\frac{\Delta_{\ell}}{\rm{Input}_\ell}$'])                                        




            fsky_binary = sum(mask) / len(mask) # only works if binary mask... 
            model_noise_Nl = get_Nl_white_noise(args, fsky_binary)

            plotTTEEBB_diff(args, cl_noise_cov_FREQ_mean[...,:3,30:]  , model_noise_Nl[...,30:cl_noise_cov_FREQ_mean.shape[-1]] *(np.pi/60/180)**0.5, '/global/u2/j/jost/Megatop/pipeline/noise_cov_vs_ModelWhiteNoise.png', 
                            legend_labels=[r'label data $D_\ell$ $\nu=$', r'label cov $D_\ell$ $\nu=$'], 
                            axis_labels=[r'$D_\ell \, [\mu K \, rad]^2$', r'$\frac{\Delta_{\ell}}{\rm{Input}_\ell}$'])        


            # noisecov_, cl_noise_cov = CheckNoiseSpectra(args, np.ones(noise_cov_mean.shape), noise_cov_mean)
            noisecov_, cl_noise_cov_preprocessed = CheckNoiseSpectra(args, np.ones(noise_cov_preprocessed_mean_pixwin.shape), 
                                                        noise_cov_preprocessed_mean_pixwin)
            plot_allspectra(args, cl_noise_cov, 'cl_noise_cov_mean.png', 
                            legend_labels=[r'label data $C_\ell$ $\nu=$', r'label model $C_\ell$ $\nu=$'], 
                            axis_labels=[r'$C_\ell \, [\mu K \, rad]^2$', r'$C_\ell \, [\mu K \, rad]^2$'])
            
            plot_allspectra(args, cl_noise_cov_preprocessed, 'cl_noise_cov_preprocessed_pixwin_mean.png', 
                            legend_labels=[r'label data $C_\ell$ $\nu=$', r'label model $C_\ell$ $\nu=$'], 
                            axis_labels=[r'$C_\ell \, [\mu K \, rad]^2$', r'$C_\ell \, [\mu K \, rad]^2$'])
            
            cl_ratio_noisecov_sim_mean = np.mean(cl_ratio_noisecov_sim_array, axis=0)

            plot_allspectra(args, cl_ratio_noisecov_sim_mean, 'test_mean.png', 
                            legend_labels=[r'label data $C_\ell$ $\nu=$', r'label model $C_\ell$ $\nu=$'], 
                            axis_labels=[r'$C_\ell \, [\mu K \, rad]^2$', r'$C_\ell \, [\mu K \, rad]^2$'])
            
            noise_cov_preprocessed_mean_cl = []
            freq_noise_cl_sim0 = []
            for f in tqdm(range(len(meta.frequencies))):
                noise_cov_preprocessed_mean_cl.append( hp.anafast(noise_cov_preprocessed_mean[f], lmax = 3*meta.general_pars['nside']) )
                freq_noise_cl_sim0.append( hp.anafast(freq_noise_maps_pre_processed_array[0][f], lmax = 3*meta.general_pars['nside']) )
            noise_cov_preprocessed_mean_cl = np.array(noise_cov_preprocessed_mean_cl)
            freq_noise_cl_sim0 = np.array(freq_noise_cl_sim0)
            
            plot_allspectra(args, noise_cov_preprocessed_mean_cl, 'noise_cov_preprocessed_mean_cl.png', 
                            legend_labels=[r'label data $C_\ell$ $\nu=$', r'label model $C_\ell$ $\nu=$'], 
                            axis_labels=[r'$C_\ell \, [\mu K \, rad]^2$', r'$C_\ell \, [\mu K \, rad]^2$'])

            plot_allspectra(args, freq_noise_cl_sim0, 'noise_cl_sim0.png', 
                            legend_labels=[r'label data $C_\ell$ $\nu=$', r'label model $C_\ell$ $\nu=$'], 
                            axis_labels=[r'$C_\ell \, [\mu K \, rad]^2$', r'$C_\ell \, [\mu K \, rad]^2$'])                            

            plotTTEEBB_diff(args, freq_noise_cl_sim0, noise_cov_preprocessed_mean_cl, '/global/u2/j/jost/Megatop/pipeline/noise_cov_preprocessed_mean_cl_diff.png', 
                            legend_labels=[r'label data $C_\ell$ $\nu=$', r'label model $C_\ell$ $\nu=$'], 
                            axis_labels=[r'$C_\ell \, [\mu K \, rad]^2$', r'$C_\ell \, [\mu K \, rad]^2$'])
            '''




        if args.plots and rank==root:
            add_test_param_in_save_name = ''
            if apply_pixwin:
                add_test_param_in_save_name +='_pixwinTrue'
           
            if not meta.noise_cov_pars['include_nhits']:
                add_test_param_in_save_name+='_nonhits'
                norm_maps = None
            else:
                norm_maps = 'hist'

                
            plot_cov_matrix(args, noise_cov_mean[:,0], 'map_noise_cov_T'+add_test_param_in_save_name+'.png', norm=norm_maps,
                            mask_unseen=binary_mask_from_nhits_nside_sims)
            plot_cov_matrix(args, noise_cov_mean[:,1], 'map_noise_cov_Q'+add_test_param_in_save_name+'.png', norm=norm_maps,
                            mask_unseen=binary_mask_from_nhits_nside_sims)
            plot_cov_matrix(args, noise_cov_mean[:,2], 'map_noise_cov_U'+add_test_param_in_save_name+'.png', norm=norm_maps,
                            mask_unseen=binary_mask_from_nhits_nside_sims)

            plot_cov_matrix(args, noise_cov_preprocessed_mean[:,0], 'map_noise_cov_preprocessed_T'+add_test_param_in_save_name+'.png', 
                            norm=norm_maps, mask_unseen=binary_mask_from_nhits_preproc)
            plot_cov_matrix(args, noise_cov_preprocessed_mean[:,1], 'map_noise_cov_preprocessed_Q'+add_test_param_in_save_name+'.png', 
                            norm=norm_maps, mask_unseen=binary_mask_from_nhits_preproc)
            plot_cov_matrix(args, noise_cov_preprocessed_mean[:,2], 'map_noise_cov_preprocessed_U'+add_test_param_in_save_name+'.png', 
                            norm=norm_maps, mask_unseen=binary_mask_from_nhits_preproc)


        if meta.noise_cov_pars['mask_to_apply']:
            mask_path = os.path.join(meta.mask_directory, meta.masks["binary_mask"])
            mask_path_before_preproc = os.path.join(meta_sims.mask_directory, meta.masks["binary_mask"])
            mask = hp.read_map(mask_path)
            mask_beforepreproc = hp.read_map(mask_path_before_preproc)
            noise_cov_mean *= mask_beforepreproc
            noise_cov_preprocessed_mean *= mask
            if args.plots:
                plot_cov_matrix(args, noise_cov_mean[:,0], 'map_noise_cov_masked_T'+add_test_param_in_save_name+'.png', 
                                mask_unseen=mask_beforepreproc, norm=norm_maps)
                plot_cov_matrix(args, noise_cov_mean[:,1], 'map_noise_cov_masked_Q'+add_test_param_in_save_name+'.png', 
                                mask_unseen=mask_beforepreproc, norm=norm_maps)
                plot_cov_matrix(args, noise_cov_mean[:,2], 'map_noise_cov_masked_U'+add_test_param_in_save_name+'.png', 
                                mask_unseen=mask_beforepreproc, norm=norm_maps)

                plot_cov_matrix(args, noise_cov_preprocessed_mean[:,0], 'map_noise_cov_preprocessed_masked_T'+add_test_param_in_save_name+'.png', 
                                mask_unseen=mask, norm=norm_maps)
                plot_cov_matrix(args, noise_cov_preprocessed_mean[:,1], 'map_noise_cov_preprocessed_masked_Q'+add_test_param_in_save_name+'.png', 
                                mask_unseen=mask, norm=norm_maps)
                plot_cov_matrix(args, noise_cov_preprocessed_mean[:,2], 'map_noise_cov_preprocessed_masked_U'+add_test_param_in_save_name+'.png', 
                                mask_unseen=mask, norm=norm_maps)

        return noise_cov_mean, noise_cov_preprocessed_mean
  
    else:
        print('sims=False: Computing noise covariance from external noise maps.')
        print('ERROR: Not yet implemented sorry...')
        exit()


def CheckWhiteNoiselvl(args, noise_cov, fsky, mask=None):
    meta = BBmeta(args.globals)

    # fsky_binary = sum(mask) / len(mask) # only works if binary mask... 

    ell, N_ell_P_SA, Map_white_noise_levels = V3.so_V3_SA_noise(
    sensitivity_mode = meta.general_pars['sensitivity_mode'],
    one_over_f_mode = 2, # fixed to None since we only use white noise here
    SAC_yrs_LF = meta.general_pars['SAC_yrs_LF'], f_sky = fsky, 
    ell_max = meta.general_pars['lmax'], delta_ell=1,
    beam_corrected=False, remove_kluge=False, CMBS4='')

    # putting masked pixels to nan to use nanmean
    if mask is not None:
        if len(mask) != noise_cov.shape[-1]:
            raise ValueError('Mask and noise_cov must have the same length')
        noise_cov_nan = noise_cov.copy()
        noise_cov_nan[...,np.where(mask==0)[0]] = np.nan
    else:
        noise_cov_nan = noise_cov

    var_noise = np.nanmean(noise_cov_nan, axis = -1) 
    std_uK_arcmin = np.sqrt(var_noise) * hp.nside2resol(hp.npix2nside(noise_cov_nan.shape[-1]), arcmin=True)

    print('std_uK_arcmin = ', std_uK_arcmin)
    print('Map_white_noise_levels = ', Map_white_noise_levels)
    print('std_uK_arcmin / Map_white_noise_levels = ', std_uK_arcmin[:,1] / Map_white_noise_levels)
    print('(std_uK_arcmin - Map_white_noise_levels) / Map_white_noise_levels * 100 = ', (std_uK_arcmin[:,1] - Map_white_noise_levels)/ Map_white_noise_levels * 100, '\n')
    return std_uK_arcmin

def CheckNoiseSpectra(args, noise_cov, noise_map, mask=None): #, remove_mean=False):
    meta = BBmeta(args.globals)

    # ratio_noisecov_sim = np.einsum('ijk,ijk->ijk',noise_map, np.sqrt(noise_cov))
    ratio_noisecov_sim = np.empty(noise_map.shape)
    cl_ratio_noisecov_sim = []

    for f in range(6):
        for s in range(3):
            ratio = noise_map[f,s] / np.sqrt(noise_cov[f,s])
            ratio[np.argwhere(np.isnan(ratio))] = 0 #regularizing the map
            # ratio_noisecov_sim.append( ratio )
            # mean_ratio = np.mean(ratio)
            ratio_noisecov_sim[f,s]  = ratio  #- mean_ratio*remove_mean
        if mask is not None:
            ratio_noisecov_sim[...,np.invert(mask)] = 0 # Maks must be binary (boolean actually)!! simply doing *= doesn't work because some value will be np.inf and np.inf*0 = np.nan which leads to problems down the line

        cl_ratio_noisecov_sim.append( hp.anafast(ratio_noisecov_sim[f], lmax = 3*meta.general_pars['nside']) )
    # ratio_noisecov_sim = np.array(ratio_noisecov_sim).reshape(6,3,-1)
    cl_ratio_noisecov_sim = np.array(cl_ratio_noisecov_sim)

    return ratio_noisecov_sim, cl_ratio_noisecov_sim


def plot_allspectra(args, Cl_data, save_name, 
                legend_labels=[r'label data $C_\ell$ $\nu=$', r'label model $C_\ell$ $\nu=$'], 
                axis_labels=['y_axis_row0', 'y_axis_row1'],
                plot_hline_one=True):
    '''
    This function plots the difference between the data and the model Cls. It directly saves the plot directly.

    Args:
        args: The parser arguments, containing the path to the global parameters file to set up metadata manager.
        Cl_data (ndarray): The data Cls, with shape (num_freq, num_spectra [TT,EE,BB], num_ell).
        Cl_model (ndarray): The model Cls, with shape (num_freq, num_spectra [TT,EE,BB], num_ell).
        save_name (str): The name of the file to save the plot. It will save the plot in the plots directory of the simulation output directory.
                            OR complete save path if you want to save it elsewhere.
        legend_labels (list): The labels for the legend of the plot.
        axis_labels (list): The labels for the x and y axes of the plot.

    Returns:
        None
    '''
    
    meta = BBmeta(args.globals)
    
    ell = np.arange(0,Cl_data.shape[-1])
    norm = ell*(ell+1)/2/np.pi
    norm /= norm

    fig, ax = plt.subplots(2,3, sharex=True, sharey='row', figsize=(15, 15))
    for f in range(Cl_data.shape[0]):
        ax[0][0].plot(ell, norm*Cl_data[f,0], 
                    color='C'+str(f),ls='-', alpha=1)
        ax[0][1].plot(ell, norm*Cl_data[f,1], 
                    color='C'+str(f),ls='-', alpha=1)
        ax[0][2].plot(ell, norm*Cl_data[f,2], label=legend_labels[0]+str(meta.general_pars['frequencies'][f]) * (Cl_data.shape[0]!=1), #
                    color='C'+str(f),ls='-', alpha=1)
        ax[1][0].plot(ell, norm*Cl_data[f,3],
                    color='C'+str(f),ls='-')
        ax[1][1].plot(ell, norm*Cl_data[f,4], 
                    color='C'+str(f),ls='-')
        ax[1][2].plot(ell, norm*Cl_data[f,5],
                    color='C'+str(f),ls='-')

    if plot_hline_one:
        ax[0][0].hlines(1,ell[0],ell[-1],linestyles='dashed',color='black')
        ax[0][1].hlines(1,ell[0],ell[-1],linestyles='dashed',color='black')
        ax[0][2].hlines(1,ell[0],ell[-1],linestyles='dashed',color='black')
        ax[1][0].hlines(1,ell[0],ell[-1],linestyles='dashed',color='black')
        ax[1][1].hlines(1,ell[0],ell[-1],linestyles='dashed',color='black')
        ax[1][2].hlines(1,ell[0],ell[-1],linestyles='dashed',color='black')

    ax[0][0].set_title('TT')
    ax[0][1].set_title('EE')
    ax[0][2].set_title('BB')
    ax[1][0].set_title('TE')
    ax[1][1].set_title('EB')
    ax[1][2].set_title('TB')        
    ax[1][0].set_xlabel('$\ell$')
    ax[1][1].set_xlabel('$\ell$')
    ax[1][2].set_xlabel('$\ell$')
    
    ax[0][0].set_ylabel(axis_labels[0])
    ax[1][0].set_ylabel(axis_labels[1])
    ax[0][2].legend(bbox_to_anchor=(1.1, 1.05), fancybox=True, shadow=True)
    # ax[0][0].loglog()
    # ax[0][1].loglog()
    # ax[0][2].loglog()

    # ax[0][0].set_yscale('log')
    # ax[0][1].set_yscale('log')
    # ax[0][2].set_yscale('log')
    # ax[1][0].set_yscale('symlog')
    # ax[1][1].set_yscale('symlog')
    # ax[1][2].set_yscale('symlog')

    ax[1][0].set_xscale('log')
    ax[1][1].set_xscale('log')
    ax[1][2].set_xscale('log')
    plt.subplots_adjust(wspace=0, hspace=0)
    
    plt.savefig( os.path.join(meta.plot_dir_from_output_dir(meta.covmat_directory_rel), save_name), bbox_inches='tight') 
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='simplistic simulator')
    parser.add_argument("--globals", type=str,
                        help="Path to yaml with global parameters")
    parser.add_argument("--use_mpi", action="store_true",
                        help="Use MPI instead of for loops to pre-process multiple maps, or simulate multiple sims.")
    parser.add_argument("--sims", default=None,
                        help="Generate a set of sims if True.")    
    parser.add_argument("--plots", action="store_true",
                        help="Plot the generated maps if True.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    noise_cov, noise_cov_preprocessed = GetNoiseCov(args)

    testing_pixwin = False
    if testing_pixwin:
        noise_cov_pixwin, noise_cov_preprocessed_pixwin = GetNoiseCov(args, apply_pixwin=True)
        relat_diff_cov = (noise_cov_pixwin - noise_cov) / np.abs(noise_cov)
        relat_diff_cov_preprocessed = (noise_cov_preprocessed_pixwin - noise_cov_preprocessed) / np.abs(noise_cov_preprocessed)
        #averaging over pixels:
        mean_relat_diff_cov = np.mean(relat_diff_cov,axis=-1) * 100
        mean_relat_diff_cov_preprocessed = np.mean(relat_diff_cov_preprocessed,axis=-1) * 100
        print('mean_relat_diff_cov = \n', mean_relat_diff_cov, '\n')
        print('mean_relat_diff_cov_preprocessed = \n', mean_relat_diff_cov_preprocessed, '\n')



    meta = BBmeta(args.globals)

    mpi = args.use_mpi
    if mpi:
        try:
            from mpi4py import MPI
            comm=MPI.COMM_WORLD
            size=comm.Get_size()
            rank=comm.rank
            barrier=comm.barrier
            root=0
            mpi = True
        except (ModuleNotFoundError, ImportError) as e:
            # Error handling
            print('ERROR IN MPI:', e)
            print('Proceeding without MPI\n')
            mpi = False
            rank=0
            pass
    else:
        rank=0 
        root = 0   
    
    if rank == root:
        # TODO: actually save inside the GetNoiseCov function? 
        print('SAVING NOISE COV')
        np.save(os.path.join(meta.output_dirs['root'], meta.output_dirs['covmat_directory'], 'pixel_noise_cov.npy' ),
                    noise_cov )
        np.save(os.path.join(meta.output_dirs['root'], meta.output_dirs['covmat_directory'], 'pixel_noise_cov_preprocessed.npy' ),
                    noise_cov_preprocessed )    

    '''
    meta_sims = BBmeta(args.sims)

    mask_path = os.path.join(meta.mask_directory, meta.masks["binary_mask"])
    mask_path_before_preproc = os.path.join(meta_sims.mask_directory, meta.masks["binary_mask"])
    mask = hp.read_map(mask_path)
    mask_beforepreproc = hp.read_map(mask_path_before_preproc)
    CheckWhiteNoiselvl(args, noise_cov, mask)
    
    # IPython.embed()

    sim_num = 0
    freq_noise_maps = np.load(os.path.join(meta_sims.output_dirs['root'], meta_sims.output_dirs['noise_directory'], 'noise_freq_maps_SIM'+str(sim_num).zfill(5)+'.npy' ) )
    ratio_noisecov_sim, cl_ratio_noisecov_sim = CheckNoiseSpectra(args, noise_cov, freq_noise_maps)
    noisecov_, cl_noise_cov = CheckNoiseSpectra(args, np.ones(noise_cov.shape), noise_cov)
    freq_noise_maps_, cl_noise_cov = CheckNoiseSpectra(args, np.ones(noise_cov.shape), noise_cov)
    plotTTEEBB_diff(args, cl_ratio_noisecov_sim, 'test.png', 
                legend_labels=[r'label data $C_\ell$ $\nu=$', r'label model $C_\ell$ $\nu=$'], 
                axis_labels=[r'$C_\ell \, [\mu K \, rad]^2$', r'$C_\ell \, [\mu K \, rad]^2$'])
    '''