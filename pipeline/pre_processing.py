import argparse
from megatop.metadata_manager import BBmeta
import IPython
import warnings
import healpy as hp
import numpy as np
from fgbuster.observation_helpers import get_instrument, get_sky, get_observation, standardize_instrument
import glob
import os
import sys
import matplotlib.pyplot as plt
import megatop.V3calc as V3
import copy
import time
import tracemalloc


def MakeSims(args):
    """This routine creates mock data from the map sets.

    Args:
        args: The parser arguments, containing the path to the global parameters file and simulation file
              to set up metadata managers.

    Returns:
        CMB_fg_noise_freq_maps (ndarray): The frequency maps of the CMB, foregrounds, and noise, with shape (num_freq, num_stokes, num_pixels).
        
        noise_maps (ndarray): The noise maps, with shape (num_freq, num_stokes, num_pixels).
        
        fg_freq_maps (ndarray): The foreground frequency maps, with shape (num_freq, num_stokes, num_pixels).
        
        cmb_sky (ndarray): The CMB sky map, with shape (num_stokes, num_pixels).
        
        CMB_fg_freq_maps_beamed (ndarray): The frequency maps of the CMB and foregrounds after beam convolution, with shape (num_freq, num_stokes, num_pixels).
        
        fsky_binary (float): The fraction of sky covered by the binary mask.

    """
    meta = BBmeta(args.globals)
    meta_sim = BBmeta(args.sims)
    d_config = meta_sim.map_sim_pars['dust_model']
    s_config = meta_sim.map_sim_pars['sync_model']

    # performing the CMB simulation with synfast
    if meta_sim.map_sim_pars['cmb_sim_no_pysm']:
        if args.verbose: print('Creating CMB map...')


        if args.plots:
            
            path_Cl_BB_lens = meta_sim.get_fname_cls_fiducial_cmb('lensed')
            path_Cl_BB_prim_r1 = meta_sim.get_fname_cls_fiducial_cmb('unlensed_scalar_tensor_r1')

            Cl_BB_prim = meta_sim.map_sim_pars['r_input']*hp.read_cl(path_Cl_BB_prim_r1)[2]
            Cl_lens = hp.read_cl(path_Cl_BB_lens)

            if args.verbose: print('Plotting Fiducial CMB spectra...')

            ell_range = np.arange(Cl_lens.shape[-1])
            Cl_prim = hp.read_cl(path_Cl_BB_prim_r1)[...,:Cl_lens.shape[-1]]

            plt.plot(ell_range, ell_range*(ell_range+1)/2/np.pi* Cl_prim[0], label='prim TT', color='C0')
            plt.plot(ell_range, ell_range*(ell_range+1)/2/np.pi* Cl_lens[0], label='lens TT', color='C0', ls='--')

            plt.plot(ell_range, ell_range*(ell_range+1)/2/np.pi* Cl_prim[1], label='prim EE', color='C1')
            plt.plot(ell_range, ell_range*(ell_range+1)/2/np.pi* Cl_lens[1], label='lens EE', color='C1', ls='--')

            plt.plot(ell_range, ell_range*(ell_range+1)/2/np.pi* Cl_prim[2], label='prim BB', color='C2')
            plt.plot(ell_range, ell_range*(ell_range+1)/2/np.pi* Cl_lens[2], label='lens BB', color='C2', ls='--')

            plt.plot(ell_range, ell_range*(ell_range+1)/2/np.pi* Cl_prim[3], label='prim TE', color='C3')
            plt.plot(ell_range, ell_range*(ell_range+1)/2/np.pi* Cl_lens[3], label='lens TE', color='C3', ls='--')

            plt.loglog()
            plt.xlabel(r'$\ell$')
            plt.ylabel(r'$D_\ell$')
            plt.legend()
            plt.savefig('fiducial_CMB_spectra.png', bbox_inches='tight')
            plt.close()



        Cl_cmb_model = get_Cl_CMB_model_from_meta(args)
        
        if meta_sim.map_sim_pars['fixed_cmb']:
            # Fixing seed so that the CMB is the same for all sims.
            # WARNING: highly wasteful as it will generate the same CMB for all sims and store them all
            # TODO: Optimize!
            np.random.seed(0) 
        cmb_sky = hp.synfast(Cl_cmb_model[0], #[Cl_TT, Cl_EE, Cl_BB, Cl_TE, Cl_EE*0.0, Cl_EE*0.0], 
                             nside=meta_sim.map_sim_pars['nside_sim'], new=True, pixwin=False)
        if meta_sim.map_sim_pars['fixed_cmb']:
            np.random.seed(None) # Resetting seed after CMB generation

    else:
        print('ERROR: CMB sims only handled using synfast on fiducial Cls')
        return 

    if args.verbose: print('Initializing Instrument ...')
    import megatop.V3calc as V3

    binary_mask = meta.read_mask('binary')        

    fsky_binary = sum(binary_mask) / len(binary_mask)

    ell, N_ell_P_SA, Map_white_noise_levels = V3.so_V3_SA_noise(
        sensitivity_mode = meta_sim.noise_sim_pars['sensitivity_mode'],
        one_over_f_mode = 2, # fixed to None since we only use white noise here
        SAC_yrs_LF = meta_sim.noise_sim_pars['SAC_yrs_LF'], f_sky = fsky_binary, 
        ell_max = meta.general_pars['lmax'], delta_ell=1,
        beam_corrected=False, remove_kluge=False, CMBS4=''
    )


    if args.verbose: print('Map_white_noise_levels = ', Map_white_noise_levels)


    instrument_config = {
        'frequency' : meta.general_pars['frequencies'],
        'depth_i' : Map_white_noise_levels/np.sqrt(2),
        'depth_p' : Map_white_noise_levels
        }

    instrument = standardize_instrument(instrument_config)


    if args.verbose: print('Creating Pysm Fg maps...')

    sky = get_sky(meta_sim.map_sim_pars['nside_sim'], d_config+s_config)

    fg_freq_maps = get_observation(instrument, sky, noise=False) 
    CMB_fg_freq_maps = fg_freq_maps + cmb_sky

    if args.verbose: print('Beaming sky maps...')
    CMB_fg_freq_maps_beamed = []


    for f in range(len(meta.general_pars['frequencies'])):

        if args.verbose: print('Beaming frequency channel:', meta.general_pars['frequencies'][f])

        lmax_convolution = 3* max(meta.general_pars['nside'], meta_sims.general_pars['nside'])


        # here lmax seems to play an important role            
        CMB_fg_alms_T, CMB_fg_alms_Q, CMB_fg_alms_U = hp.map2alm(CMB_fg_freq_maps[f], lmax=lmax_convolution, pol=True)

        Bl_gauss_fwhm = hp.gauss_beam( np.radians(meta.pre_proc_pars['fwhm'][f]/60), lmax=lmax_convolution, pol=True)
        
        wpix_in = hp.pixwin( meta_sims.general_pars['nside'], pol=True, lmax=lmax_convolution) # Pixel window function of input maps
          
        sm_corr_T =  Bl_gauss_fwhm[:,0] * wpix_in[0]
        sm_corr_P =  Bl_gauss_fwhm[:,1] * wpix_in[1]

        #change beam and wpix
        alm_out_T = hp.almxfl(CMB_fg_alms_T, sm_corr_T)
        alm_out_E = hp.almxfl(CMB_fg_alms_Q, sm_corr_P)
        alm_out_B = hp.almxfl(CMB_fg_alms_U, sm_corr_P)

        #alm-->mapf
        CMB_fg_alms_out_T, CMB_fg_alms_out_Q, CMB_fg_alms_out_U = hp.alm2map([alm_out_T,alm_out_E,alm_out_B], meta_sims.general_pars['nside'],
                                                    lmax=lmax_convolution, pixwin=False, fwhm=0.0, pol=True) 


        CMB_fg_freq_maps_beamed.append([CMB_fg_alms_out_T, CMB_fg_alms_out_Q, CMB_fg_alms_out_U])        
    CMB_fg_freq_maps_beamed = np.array(CMB_fg_freq_maps_beamed)

    
    if args.verbose: print('Creating noise maps...')

    if meta_sim.noise_sim_pars['noise_option']=='white_noise':
        nlev_map = fg_freq_maps*0.0
        for f in range(len(instrument.frequency)):
            nlev_map[f] = np.array([instrument.depth_i[f], instrument.depth_p[f], instrument.depth_p[f]])[:,np.newaxis]*np.ones((3,fg_freq_maps.shape[-1]))
        nlev_map /= hp.nside2resol(meta_sim.map_sim_pars['nside_sim'], arcmin=True)
        noise_maps = np.random.normal(fg_freq_maps*0.0, nlev_map, fg_freq_maps.shape)
    elif meta_sim.noise_sim_pars['noise_option']=='':

        if args.verbose: print('No noise case')

        noise_maps = 0 * fg_freq_maps
    else:
        print('ERROR: Other noise cases not handled yet...')
        return
 
 
    if meta_sim.noise_sim_pars['noise_option'] != '' and meta_sim.noise_sim_pars['include_nhits']:
        if args.verbose: print('Including nhits in noise maps...')
        nhits_map = meta_sim.read_hitmap() 
        nhits_map_rescaled = nhits_map / max(nhits_map)
        binary_mask_sim = meta_sim.read_mask('binary')

        warnings.filterwarnings("error")        
        try:
            noise_maps[...,np.where(binary_mask_sim==1)[0]] /= np.sqrt(nhits_map_rescaled[np.where(binary_mask_sim==1)[0]])
            # This avoids dividing by 0 in the noise maps
        except RuntimeWarning:
            print('ERROR: Division by 0 in noise map nhit rescaling.')
            print('This means the binary mask is not covering all the parts where nhits = 0.')
            print('Please check the mask_handling parameters, changing "mask_handler_binary_zero_threshold" can help.')
            print('Exiting...')
            exit()
        warnings.resetwarnings()
        noise_maps[...,np.where(binary_mask_sim==0)[0]] = 0 # hp.UNSEEN

    
    CMB_fg_noise_freq_maps = CMB_fg_freq_maps_beamed + noise_maps


    if meta_sim.noise_sim_pars['include_nhits']:
        # Importing mask from the simulations nside to apply to simulated maps
        # Necessary to do it here when applying 1/sqrt(nhits) to the noise maps as it will create inf in the noise maps
        binary_mask_sim = meta_sim.read_mask('binary')
        CMB_fg_noise_freq_maps[...,np.where(binary_mask_sim==0)[0]] = 0 # hp.UNSEEN


    return CMB_fg_noise_freq_maps, noise_maps, fg_freq_maps, cmb_sky, CMB_fg_freq_maps_beamed, fsky_binary
        


def get_maps(args):
    meta = BBmeta(args.globals)
    # get path from maps and import them
    print('DUMMY FUNCTION RETURN MAPS FULL OF ONES FOR TESTING NSIDE_INPUT = 512')
    print('SHAPE = (NFREQ, NSTOKES, NPIX)')
    freq_maps = np.ones((6,3,hp.nside2npix(512)))
    # freq_maps = np.ones((6,3,hp.nside2npix(meta.general_pars['nside'])))
    return freq_maps

def CommonBeamConvAndNsideModification(args, freq_maps):
    '''
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
    '''

    meta = BBmeta(args.globals)
    map_dimensions = len(freq_maps.shape)


    freq_maps_out = []

    if args.verbose: print('  -> common beam correction and NSIDE change: correcting for frequency-dependent beams, convolving with a common beam, modifying NSIDE and include effect of pixel window function')


    lmax_convolution = 3*meta.general_pars['nside']
    wpix_in = hp.pixwin( hp.npix2nside(freq_maps.shape[-1]),pol=True,lmax=lmax_convolution) # Pixel window function of input maps
    wpix_out = hp.pixwin(meta.general_pars['nside'],pol=True,lmax=lmax_convolution) # Pixel window function of output maps
    wpix_in[1][0:2] = 1. #in order not to divide by 0
    Bl_gauss_common = hp.gauss_beam(np.radians(meta.pre_proc_pars['common_beam_correction']/60), lmax=lmax_convolution, pol=True)

 
    for f in range(len(meta.general_pars['frequencies'])):
        #beam corrections
        Bl_gauss_fwhm = hp.gauss_beam( np.radians(meta.pre_proc_pars['fwhm'][f]/60), lmax=lmax_convolution, pol=True)

        bl_correction =  Bl_gauss_common / Bl_gauss_fwhm


        sm_corr_T = bl_correction[:,0] * wpix_out[0]/wpix_in[0]
        sm_corr_P = bl_correction[:,1] * wpix_out[1]/wpix_in[1]

        #map-->alm
        if map_dimensions == 2: # if maps are stored in (nstokes*nfreq, npix)
            cmb_in_T,cmb_in_Q,cmb_in_U = freq_maps[3*f:3*(f+1),:]
        elif map_dimensions == 3:
            cmb_in_T,cmb_in_Q,cmb_in_U = freq_maps[f]
        else:
            print('freq_maps doesn\'t have the right number of dimensions, either 2 (nstokes*nfreq, npix), or 3 (nfreq, nstokes, npix)') 
            print('returning original freq_maps ...')
            return freq_maps
        
        alm_in_T,alm_in_E,alm_in_B = hp.map2alm([cmb_in_T,cmb_in_Q,cmb_in_U],lmax=lmax_convolution,pol=True, iter=10)
        # here lmax seems to play an important role            

        #change beam and wpix
        alm_out_T = hp.almxfl(alm_in_T,sm_corr_T)
        alm_out_E = hp.almxfl(alm_in_E,sm_corr_P)
        alm_out_B = hp.almxfl(alm_in_B,sm_corr_P)

        #alm-->mapf
        cmb_out_T,cmb_out_Q,cmb_out_U = hp.alm2map([alm_out_T,alm_out_E,alm_out_B],meta.general_pars['nside'],
                                                    lmax=lmax_convolution,pixwin=False,fwhm=0.0,pol=True) 

        # a priori all the options are set to there default, even lmax which is computed wrt input alms
        marco_out_map = np.array([cmb_out_T,cmb_out_Q,cmb_out_U])
        freq_maps_out.append(marco_out_map)
    
    return np.array(freq_maps_out)

def ApplyBinaryMask(args, freq_maps, use_UNSEEN = False):
    '''
    This function applies the binary mask to the frequency maps.

    Args:
        args: The parser arguments, containing the path to the global parameters file to set up metadata manager.
        freq_maps (ndarray): The frequency maps to mask, with shape (num_freq, num_stokes, num_pixels).
        use_UNSEEN (bool): If True, the UNSEEN value is used for the masked pixels, otherwise the masked pixels are set to zero.

    Returns:
        freq_maps_masked (ndarray): The frequency maps after applying the binary mask, with shape (num_freq, num_stokes, num_pixels).
    '''
    meta = BBmeta(args.globals)
    binary_mask_path = meta.get_fname_mask('binary')

    binary_mask = hp.read_map(
        binary_mask_path,
        dtype=float)

    freq_maps_masked = copy.deepcopy(freq_maps)

    if use_UNSEEN:
        freq_maps_masked[:,np.where(binary_mask==0)[0]] = hp.UNSEEN

    else:
        freq_maps_masked *= binary_mask

    return freq_maps_masked

def plotTTEEBB_diff(args, Cl_data, Cl_model, save_name, 
                    legend_labels=[r'label data $C_\ell$ $\nu=$', r'label model $C_\ell$ $\nu=$'], 
                    axis_labels=['y_axis_row0', 'y_axis_row1'],
                    use_D_ell = True):

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
        meta_sims = BBmeta(args.sims)
        meta = BBmeta(args.globals)
        
        ell = np.arange(0,Cl_data.shape[-1])
        norm = ell*(ell+1)/2/np.pi

        if not use_D_ell:
            norm = 1
        
        fig, ax = plt.subplots(2,3, sharex=True, sharey='row', figsize=(15, 15))
        for f in range(Cl_data.shape[0]):
            ax[0][0].plot(ell, norm*Cl_data[f,0], 
                       color='C'+str(f),ls='-', alpha=0.4)
            ax[0][1].plot(ell, norm*Cl_data[f,1], 
                       color='C'+str(f),ls='-', alpha=0.4)
            ax[0][2].plot(ell, norm*Cl_data[f,2], label=legend_labels[0]+str(meta.general_pars['frequencies'][f]) * (Cl_data.shape[0]!=1), #
                       color='C'+str(f),ls='-', alpha=0.4)
            ax[0][0].plot(ell, norm*Cl_model[f,0],
                       color='C'+str(f),ls=':')
            ax[0][1].plot(ell, norm*Cl_model[f,1], 
                       color='C'+str(f),ls=':')
            ax[0][2].plot(ell, norm*Cl_model[f,2], label=legend_labels[1]+str(meta.general_pars['frequencies'][f]) * (Cl_data.shape[0]!=1),  #
                       color='C'+str(f),ls=':')

            zero_index_model0 = np.where(Cl_model[f,0] != 0)[0]
            zero_index_model1 = np.where(Cl_model[f,1] != 0)[0]
            zero_index_model2 = np.where(Cl_model[f,2] != 0)[0]
            ax[1][0].plot(ell[zero_index_model0], ((Cl_data[f,0] - Cl_model[f,0])[zero_index_model0]/Cl_model[f,0,zero_index_model0]), 
                       color='C'+str(f),ls='-', alpha=0.4)
            ax[1][1].plot(ell[zero_index_model1], ((Cl_data[f,1] - Cl_model[f,1])[zero_index_model1]/Cl_model[f,1,zero_index_model1]), 
                       color='C'+str(f),ls='-', alpha=0.4)
            ax[1][2].plot(ell[zero_index_model2], ((Cl_data[f,2] - Cl_model[f,2])[zero_index_model2]/Cl_model[f,2,zero_index_model2]),
                       color='C'+str(f),ls='-', alpha=0.4)             

        ax[0][0].set_title('TT')
        ax[0][1].set_title('EE')
        ax[0][2].set_title('BB')
        ax[1][0].set_xlabel(r'\ell')
        ax[1][1].set_xlabel(r'\ell')
        ax[1][2].set_xlabel(r'\ell')
        ax[0][0].set_ylabel(axis_labels[0])
        ax[1][0].set_ylabel(axis_labels[1])
        ax[0][2].legend(bbox_to_anchor=(1.1, 1.05), fancybox=True, shadow=True)
        ax[0][0].loglog()
        ax[0][1].loglog()
        ax[0][2].loglog()
        ax[1][0].set_xscale('log')
        ax[1][1].set_xscale('log')
        ax[1][2].set_xscale('log')
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.savefig(os.path.join(meta_sims.plots_directory, save_name), bbox_inches='tight') 
        plt.close()

def get_Cl_CMB_model_from_meta(args):
    '''
    This function reads the fiducial CMB Cls from the metadata manager and combines scalar, lensing and tensor 
    contributions to return the model Cls according to A_lens and r in the simulation parameter file.

    Args:
        args: The parser arguments, containing the path to the global parameters file to set up metadata manager.
        
    Returns:
        Cl_cmb_model (ndarray): The model CMB Cls, with shape (num_freq, num_spectra [TT,EE,BB,TE,EB,TB], num_ell).
    '''
    meta_sims = BBmeta(args.sims)
    path_Cl_BB_lens = meta_sims.get_fname_cls_fiducial_cmb('lensed')
    path_Cl_BB_prim_r1 = meta_sims.get_fname_cls_fiducial_cmb('unlensed_scalar_tensor_r1')

    Cl_BB_prim = meta_sims.map_sim_pars['r_input']*hp.read_cl(path_Cl_BB_prim_r1)[2]
    Cl_lens = hp.read_cl(path_Cl_BB_lens)

    l_max_lens = len(Cl_lens[0])
    Cl_BB_lens = meta_sims.map_sim_pars['A_lens']*Cl_lens[2]
    Cl_TT = Cl_lens[0]
    Cl_EE = Cl_lens[1]
    Cl_TE = Cl_lens[3]

    Cl_BB = Cl_BB_prim[:l_max_lens] + Cl_BB_lens
    Cl_cmb_model = np.array([[Cl_TT, Cl_EE, Cl_BB, Cl_TE, Cl_EE*0.0, Cl_EE*0.0]])    
    return Cl_cmb_model

def get_Nl_white_noise(args, fsky_binary):
    '''
    
    This function computes the white noise level from V3calc using parameters from the metadata manager 
    and returns the noise Cls.


    Args:
        args: The parser arguments, containing the path to the global parameters file to set up metadata manager.
        fsky_binary (float): The fraction of sky covered by the binary mask.
        
    Returns:
        model_noise (ndarray): The model noise Cls, with shape (num_freq, num_spectra [TT,EE,BB], num_ell).
    '''
    meta_sims = BBmeta(args.sims)
    meta = BBmeta(args.globals)

    ell_V3, N_ell_P_SA, Map_white_noise_levels = V3.so_V3_SA_noise(
            sensitivity_mode = meta_sims.noise_sim_pars['sensitivity_mode'],
            one_over_f_mode = 2, # fixed to None since we only use white noise here
            SAC_yrs_LF = meta_sims.noise_sim_pars['SAC_yrs_LF'], f_sky = fsky_binary, 
            ell_max = meta.general_pars['lmax'], delta_ell=1,
            beam_corrected=False, remove_kluge=False, CMBS4='' )
            
    if args.verbose: print('Map_white_noise_levels = ', Map_white_noise_levels)


    lmax_convolution = 3*meta_sims.general_pars['nside']

    N_ell_white_f_arcmin = []
    for f in range(len(meta.general_pars['frequencies'])):
        N_ell_white_f_arcmin.append(np.ones(lmax_convolution) *
                                Map_white_noise_levels[f]**2)
    N_ell_white_f = np.array(N_ell_white_f_arcmin)  * (np.pi/60/180)**2 
    N_ell_white_f_temp = np.array(N_ell_white_f_arcmin)  * (np.pi/60/180)**2 /2

    model_noise = np.empty([len(meta.general_pars['frequencies']), 3, lmax_convolution])
    model_noise[:,0] = N_ell_white_f_temp
    model_noise[:,1] = N_ell_white_f
    model_noise[:,2] = N_ell_white_f  

    return model_noise

def check_sims(args, cmb_sky, noise_maps, freq_maps, fg_freq_maps, CMB_fg_freq_maps_beamed,fsky_binary):
        '''	    
        This function checks the simulated maps by comparing their Cls with the input CMB spectra beamed, the foreground Cls computed by their input map beamed
        and the white noise Cl. It saves the different plots in the plots directory of the simulation output directory.
        
        Args:
            args: The parser arguments, containing the path to the global parameters file to set up metadata manager.
            cmb_sky (ndarray): The CMB sky map, with shape (num_stokes, num_pixels).
            noise_maps (ndarray): The noise maps, with shape (num_freq, num_stokes, num_pixels).
            freq_maps (ndarray): The frequency maps, with shape (num_freq, num_stokes, num_pixels).
            fg_freq_maps (ndarray): The foreground frequency maps, with shape (num_freq, num_stokes, num_pixels).
            CMB_fg_freq_maps_beamed (ndarray): The frequency maps of the CMB and foregrounds after beam convolution, with shape (num_freq, num_stokes, num_pixels).
            fsky_binary (float): The fraction of sky covered by the binary mask.
        
        Returns:
            None
        '''

        meta_sim = BBmeta(args.sims)
        meta = BBmeta(args.globals)
        cl_cmb_sky_maps = hp.anafast(cmb_sky)

        cl_noise_f = []
        cl_freq_maps_f = []
        cl_fg_freq_maps = []
        cl_CMB_fg_freq_maps_beamed = []

        for f in range(len(meta.general_pars['frequencies'])):

            cl_noise_f.append( hp.anafast(noise_maps[f]) )

            cl_freq_maps_f.append( hp.anafast(freq_maps[f]))
            cl_fg_freq_maps.append(hp.anafast(fg_freq_maps[f]))
            cl_CMB_fg_freq_maps_beamed.append(hp.anafast(CMB_fg_freq_maps_beamed[f]))
        cl_noise_f = np.array(cl_noise_f)
        cl_freq_maps_f = np.array(cl_freq_maps_f)
        cl_fg_freq_maps = np.array(cl_fg_freq_maps)
        cl_CMB_fg_freq_maps_beamed = np.array(cl_CMB_fg_freq_maps_beamed)

        model_noise = get_Nl_white_noise(args, fsky_binary)


        plotTTEEBB_diff(args, cl_noise_f * fsky_binary, model_noise, 
                        os.path.join(meta_sims.plots_directory, 'Noise_lvl_check.png'), 
                        legend_labels=[r'Noise $C_\ell$ from map $\nu=$', r'Input white noise lvl $\nu=$'], 
                        axis_labels=[r'$D_\ell \, [\mu K \, rad]^2$', r'$\frac{\Delta_{\ell}}{\rm{Input}_\ell}$'])

        Cl_cmb_model = get_Cl_CMB_model_from_meta(args)


        plotTTEEBB_diff(args, np.array([cl_cmb_sky_maps])* fsky_binary, Cl_cmb_model[...,:cl_cmb_sky_maps.shape[-1]] * fsky_binary, 
                        os.path.join(meta_sims.plots_directory, 'CMB_check.png'), 

                        legend_labels=[r'$C_{\ell}^{\rm CMB}$ from map', r'Input $C_{\ell}^{\rm CMB}$'], 
                        axis_labels=[r'$D_\ell \, [\mu K \, rad]^2$', r'$\frac{\Delta_{\ell}}{\rm{Input}_\ell}$'])  

        Bl_gauss_fwhm_freq = []
        for f in range(len(meta.general_pars['frequencies'])):
            Bl_gauss_fwhm_freq.append( hp.gauss_beam( np.radians(meta.pre_proc_pars['fwhm'][f]/60), lmax=3*meta_sim.map_sim_pars['nside_sim']-1, pol=True))
        Bl_gauss_fwhm_freq = np.array(Bl_gauss_fwhm_freq)

        wpix_in = hp.pixwin( meta_sims.general_pars['nside'], pol=True, lmax=3*meta_sim.map_sim_pars['nside_sim']-1) # Pixel window function of input maps

        beamed_sky_TT = (Cl_cmb_model[...,:3*meta_sim.map_sim_pars['nside_sim']] + cl_fg_freq_maps)[:,0] * Bl_gauss_fwhm_freq[...,0]**2 * wpix_in[0]**2
        beamed_sky_EE = (Cl_cmb_model[...,:3*meta_sim.map_sim_pars['nside_sim']] + cl_fg_freq_maps)[:,1] * Bl_gauss_fwhm_freq[...,1]**2 * wpix_in[1]**2
        beamed_sky_BB = (Cl_cmb_model[...,:3*meta_sim.map_sim_pars['nside_sim']] + cl_fg_freq_maps)[:,2] * Bl_gauss_fwhm_freq[...,1]**2 * wpix_in[1]**2

        beamed_sky_model = np.array([ beamed_sky_TT, beamed_sky_EE, beamed_sky_BB ]).swapaxes(0,1)


        plotTTEEBB_diff(args, cl_CMB_fg_freq_maps_beamed * fsky_binary, beamed_sky_model * fsky_binary, 
                        os.path.join(meta_sims.plots_directory, 'sky_beamed_check.png') , 
                        legend_labels=[r'$C_{\ell}^{\rm CMB+fg beamed}$ from map', r'Input $C_{\ell}^{\rm CMB+fg beamed}$'], 
                        axis_labels=[r'$D_\ell \, [\mu K \, rad]^2$', r'$\frac{\Delta_{\ell}}{\rm{Input}_\ell}$'])

def check_preproc(args, preproc_freq_maps, fg_freq_maps, fsky_binary, sim_num=0):
    '''
    
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
    '''
    meta_sims = BBmeta(args.sims)
    meta = BBmeta(args.globals)

    Cl_cmb_model = get_Cl_CMB_model_from_meta(args)[:,:3] # Keeping only TT, EE, BB

    cl_fg_freq_maps = []
    cl_preproc_freq_maps = []

    for f in range(len(meta.general_pars['frequencies'])):
        cl_fg_freq_maps.append( hp.anafast(fg_freq_maps[f]))
        cl_preproc_freq_maps.append( hp.anafast(preproc_freq_maps[f]))
    cl_fg_freq_maps = np.array(cl_fg_freq_maps)[:,:3] # Keeping only TT, EE, BB
    cl_preproc_freq_maps = np.array(cl_preproc_freq_maps)[:,:3] # Keeping only TT, EE, BB

    lmax_convolution = 3*meta.general_pars['nside']
    wpix_in = hp.pixwin( meta_sims.general_pars['nside'],pol=True,lmax=lmax_convolution) # Pixel window function of input maps
    wpix_out = hp.pixwin(meta.general_pars['nside'],pol=True,lmax=lmax_convolution) # Pixel window function of output maps
    wpix_in[1][0:2] = 1. #in order not to divide by 0
    Bl_gauss_common = hp.gauss_beam(np.radians(meta.pre_proc_pars['common_beam_correction']/60), lmax=lmax_convolution, pol=True)
    
    beam_correction = []
    for f in range(len(meta.general_pars['frequencies'])):
        #beam corrections
        Bl_gauss_fwhm = hp.gauss_beam( np.radians(meta.pre_proc_pars['fwhm'][f]/60), lmax=lmax_convolution, pol=True)

        bl_correction =  Bl_gauss_common / Bl_gauss_fwhm

        sm_corr_T = bl_correction[:,0] * wpix_out[0]/wpix_in[0]
        sm_corr_P = bl_correction[:,1] * wpix_out[1]/wpix_in[1]

        beam_correction.append([sm_corr_T, sm_corr_P, sm_corr_P])
    beam_correction = np.array(beam_correction)

    beam_correction_NODECONV_T = Bl_gauss_common[:,0] * wpix_out[0] #/wpix_in[0]
    beam_correction_NODECONV_P = Bl_gauss_common[:,1] * wpix_out[1] #/wpix_in[1]
    beam_correction_NODECONV = np.array([beam_correction_NODECONV_T, beam_correction_NODECONV_P, beam_correction_NODECONV_P])

    # CMB spectra and fg maps (and their spectra) are not convolved by their respective frequency beams
    # So to mimic and fit the pre-processed maps, only the common beams and pixel window functions must be applied.
    CMB_fg_beamed_spectra = (Cl_cmb_model[...,:lmax_convolution] + cl_fg_freq_maps[...,:lmax_convolution]) * np.array([beam_correction_NODECONV[...,:lmax_convolution]**2])


    # Noise after pre-processing model: 
    model_noise = get_Nl_white_noise(args, fsky_binary)
    model_noise_beamed = model_noise[...,:lmax_convolution] * beam_correction[...,:lmax_convolution]**2

    model_beamed_total = model_noise_beamed + CMB_fg_beamed_spectra

    plotTTEEBB_diff(args, cl_preproc_freq_maps, model_beamed_total, os.path.join( meta.plot_dir_from_output_dir(meta.pre_process_directory_rel), 'preproc_check_SIM'+str(sim_num).zfill(5)+'.png'), 
                    legend_labels=[r'Preproc $C_\ell$ from map $\nu=$', r'Model Cl after preproc $\nu=$'], 
                    axis_labels=[r'$D_\ell \, [\mu K \, rad]^2$', r'$\frac{\Delta_{\ell}}{\rm{Input}_\ell}$'])    
    
    
    np.save(os.path.join(meta_sims.comb_spectra_directory, 'spectra_comb_freq_maps_SIM'+str(sim_num).zfill(5)+'.npy' ), cl_preproc_freq_maps )

    return cl_preproc_freq_maps, model_beamed_total

def wrapper_MakeSimsMPI(args, rank, size):
    '''
    This function is a wrapper for the MakeSims function, to be used in MPI mode.
    It is used to call MakeSims with the correct simulation number in MPI mode.

    Args:
        args: The parser arguments, containing the path to the global parameters file to set up metadata manager.
        rank (int): The rank of the MPI process.
        size (int): The total number of MPI processes.

    Returns:
        freq_maps: The frequency maps, with shape (num_freq, num_stokes, num_pixels).
        fsky_binary: The fraction of sky covered by the binary mask.
        fg_freq_maps: The foreground frequency maps, with shape (num_freq, num_stokes, num_pixels).
                      Only used when checking the prepocessing step via plotting.

    '''
    meta_sims = BBmeta(args.sims)

    if meta_sims.general_pars['nsims'] != size:
        exit('ERROR: nsims must be equal to size in MPI mode. nsims = '+ str(meta_sims.general_pars['nsims'])+'  size = '+ str(size))

    if args.verbose: print('simulating maps sim number: ', rank + 1, '/', meta_sims.general_pars['nsims'], ' (MPI)')
    if args.verbose: print('Memory usage before simulation is: ', tracemalloc.get_traced_memory())
    freq_maps, noise_maps, fg_freq_maps, cmb_sky, CMB_fg_freq_maps_beamed, fsky_binary = MakeSims(args)
    if args.verbose: print('Memory usage after simulation is: ', tracemalloc.get_traced_memory())
    np.save(os.path.join(meta_sims.comb_directory, 'comb_freq_maps_SIM'+str(rank).zfill(5)+'.npy' ), 
            freq_maps )
    np.save(os.path.join(meta_sims.cmb_directory, 'cmb_maps_SIM'+str(rank).zfill(5)+'.npy' ), 
            cmb_sky )
    np.save(os.path.join(meta_sims.cmb_beamed_directory, 'cmb_beamed_freq_maps_SIM'+str(rank).zfill(5)+'.npy' ),
                CMB_fg_freq_maps_beamed )
    np.save(os.path.join(meta_sims.fg_directory, 'fg_freq_maps_SIM'+str(rank).zfill(5)+'.npy' ), 
            fg_freq_maps )
    np.save(os.path.join(meta_sims.noise_directory, 'noise_freq_maps_SIM'+str(rank).zfill(5)+'.npy' ), 
            noise_maps )   

    if args.plots:
        if args.verbose:print('checking sims...')
        # This only checks the last simulations
        # TODO: in MPI do all cases or just one? 
        if args.verbose: print('Memory usage before checking sims is: ', tracemalloc.get_traced_memory())
        if rank==0:
            check_sims(args, cmb_sky, noise_maps, freq_maps, fg_freq_maps, CMB_fg_freq_maps_beamed, fsky_binary)
        if args.verbose: print('Memory usage after checking sims is: ', tracemalloc.get_traced_memory())
                    

    return freq_maps, fsky_binary, fg_freq_maps


def wrapper_MakeSimsNOTMPI(args):
    '''
    This function is a wrapper for the MakeSims function, to be used in non-MPI mode.
    It is used to call MakeSims in a for loop for each simulation number.

    Args:
        args: The parser arguments, containing the path to the global parameters file to set up metadata manager.

    Returns:
        freq_maps_sim_list: A list of the frequency maps for each simulation, with shape (num_sims, num_freq, num_stokes, num_pixels).
        fsky_binary: The fraction of sky covered by the binary mask.
        fg_freq_maps: The foreground frequency maps, with shape (num_freq, num_stokes, num_pixels).
                      Only used when checking the prepocessing step via plotting. 
    '''
    meta_sims = BBmeta(args.sims)

    freq_maps_sim_list = []
    for sim_num in range(meta_sims.general_pars['nsims']):

        if args.verbose: print('simulating maps sim number: ', sim_num + 1, '/', meta_sims.general_pars['nsims'], ' (for loop, NOT MPI)')
        if args.verbose: print('Memory usage before simulation is: ', tracemalloc.get_traced_memory())
        
        freq_maps, noise_maps, fg_freq_maps, cmb_sky, CMB_fg_freq_maps_beamed, fsky_binary = MakeSims(args)
        
        if args.verbose: print('Memory usage after simulation is: ', tracemalloc.get_traced_memory())
        
        freq_maps_sim_list.append(freq_maps)
        
        if args.verbose: print('saving map sims ...')
        np.save(os.path.join(meta_sims.comb_directory, 'comb_freq_maps_SIM'+str(sim_num).zfill(5)+'.npy' ),
                    freq_maps )
        np.save(os.path.join(meta_sims.cmb_directory, 'cmb_maps_SIM'+str(sim_num).zfill(5)+'.npy' ), 
                cmb_sky )
        np.save(os.path.join(meta_sims.cmb_beamed_directory, 'cmb_beamed_freq_maps_SIM'+str(sim_num).zfill(5)+'.npy' ), 
                CMB_fg_freq_maps_beamed )
        np.save(os.path.join(meta_sims.fg_directory, 'fg_freq_maps_SIM'+str(sim_num).zfill(5)+'.npy' ), 
                fg_freq_maps )
        np.save(os.path.join(meta_sims.noise_directory, 'noise_freq_maps_SIM'+str(sim_num).zfill(5)+'.npy' ), 
                noise_maps )    
    
    if args.plots:
        if args.verbose:print('checking sims...')
        # This only checks the last simulations
        if args.verbose: print('Memory usage before checking sims is: ', tracemalloc.get_traced_memory())
        check_sims(args, cmb_sky, noise_maps, freq_maps, fg_freq_maps, CMB_fg_freq_maps_beamed, fsky_binary)
        if args.verbose: print('Memory usage after checking sims is: ', tracemalloc.get_traced_memory())
                    

    return freq_maps_sim_list, fsky_binary, fg_freq_maps


def wrapper_preprocNOTMPI(args, freq_maps_sim_list, fg_freq_maps=None, fsky_binary=None):
    '''
    This function is a wrapper for the pre-processing functions, to be used in non-MPI mode.
    It is used to call the pre-processing functions in a for loop for each simulation number.

    Args:
        args: The parser arguments, containing the path to the global parameters file to set up metadata manager.
        freq_maps_sim_list: A list of the frequency maps for each simulation, with shape (num_sims, num_freq, num_stokes, num_pixels).
        fg_freq_maps (ndarray): The foreground frequency maps, with shape (num_freq, num_stokes, num_pixels). Only used if args.sims and args.plots are True.
        fsky_binary (float): The fraction of sky covered by the binary mask. Only used if args.sims and args.plots are True.
    Returns:
        None
    '''
    meta_sims = BBmeta(args.sims)

    for sim_num in range(meta_sims.general_pars['nsims']):
        if args.verbose: print('Memory usage before pre-processing is: ', tracemalloc.get_traced_memory())
        if args.verbose: print('Pre-precessing freq-maps #', sim_num + 1, ' out of ', meta_sims.general_pars['nsims'])
        freq_maps_common_beamed = CommonBeamConvAndNsideModification(args, freq_maps_sim_list[sim_num])
        freq_maps_common_beamed_masked = ApplyBinaryMask(args, freq_maps_common_beamed)
        if args.verbose: print('Memory usage after pre-processing is: ', tracemalloc.get_traced_memory())
        if args.verbose: print('saving pre-processed maps ...')
        np.save(os.path.join(meta.pre_process_directory, 'freq_maps_common_beamed_masked'+str(sim_num).zfill(5)+'.npy' ),
                freq_maps_common_beamed_masked )
        np.save(os.path.join(meta.pre_process_directory, 'freq_maps_common_beamed'+str(sim_num).zfill(5)+'.npy' ),
                freq_maps_common_beamed )

    if args.sims and args.plots:
        if args.verbose: print('checking pre-processed maps...\n')
        # This only checks the last simulations
        cl_preproc_freq_maps, model_beamed_total = check_preproc( args, freq_maps_common_beamed, fg_freq_maps, fsky_binary, sim_num=sim_num)


def wrapper_preprocMPI(args, freq_maps, rank, size,  fg_freq_maps=None, fsky_binary=None):
    '''
    This function is a wrapper for the pre-processing functions, to be used in MPI mode.
    It is used to call the pre-processing functions with the correct simulation number in MPI mode.

    Args:
        args: The parser arguments, containing the path to the global parameters file to set up metadata manager.
        freq_maps: The frequency maps, with shape (num_freq, num_stokes, num_pixels).
        rank (int): The rank of the MPI process.
        size (int): The total number of MPI processes.
        fg_freq_maps (ndarray): The foreground frequency maps, with shape (num_freq, num_stokes, num_pixels). Only used if args.sims and args.plots are True.
        fsky_binary (float): The fraction of sky covered by the binary mask. Only used if args.sims and args.plots are True.

    
    Returns:
        None

    '''
    meta_sims = BBmeta(args.sims)

    if args.verbose: print('Pre-precessing freq-maps #', rank + 1, ' out of ', size, ' (MPI)')
    if args.verbose: print('Memory usage before pre-processing is: ', tracemalloc.get_traced_memory())
    freq_maps_common_beamed = CommonBeamConvAndNsideModification(args, freq_maps)
    freq_maps_common_beamed_masked = ApplyBinaryMask(args, freq_maps_common_beamed)
    if args.verbose: print('Memory usage after pre-processing is: ', tracemalloc.get_traced_memory())

    if args.verbose: print('saving pre-processed maps ...\n')
    np.save(os.path.join(meta.pre_process_directory, 'freq_maps_common_beamed_masked'+str(rank).zfill(5)+'.npy' ),
            freq_maps_common_beamed_masked )
    
    if not meta_sims.noise_sim_pars['include_nhits']:
        # If nhits is used for the noise, all maps should be masked. 
        np.save(os.path.join(meta.pre_process_directory, 'freq_maps_common_beamed'+str(rank).zfill(5)+'.npy' ),
                freq_maps_common_beamed )
        
    if args.sims and args.plots:
        if args.verbose: print('checking pre-processed maps...\n')
        if args.verbose: print('Memory usage before checking pre-processed maps is: ', tracemalloc.get_traced_memory())
        cl_preproc_freq_maps, model_beamed_total = check_preproc( args, freq_maps_common_beamed, fg_freq_maps, fsky_binary, sim_num=rank)        
        if args.verbose: print('Memory usage after checking pre-processed maps is: ', tracemalloc.get_traced_memory())
        # Ensure recvbuf is contiguous
        # to make sure the comm.Gather() works correctly

        cl_preproc_freq_maps = np.ascontiguousarray(cl_preproc_freq_maps) 

        recvbuf = None
        if rank == 0:
            shape_recvbuf = (size,) + cl_preproc_freq_maps.shape
            recvbuf = np.empty(shape_recvbuf)

            # Ensure recvbuf is contiguous
            # to make sure the comm.Gather() works correctly
            recvbuf = np.ascontiguousarray(recvbuf)

        comm.Gather(cl_preproc_freq_maps, recvbuf, root=0)

        if rank == 0:
            if args.verbose: print('checking MEAN preproc results...\n')

            mean_cl_preproc_freq_maps = np.mean(recvbuf, axis=0)
            plotTTEEBB_diff(args, mean_cl_preproc_freq_maps, model_beamed_total,
                            os.path.join( meta.plot_dir_from_output_dir(meta.pre_process_directory_rel), 'mean_preproc_cl_check.png'), 
                            legend_labels=[r'Mean preproc $C_\ell$ from map $\nu=$', r'Model Cl after preproc $\nu=$'], 
                            axis_labels=[r'$D_\ell \, [\mu K \, rad]^2$', r'$\frac{\Delta_{\ell}}{\rm{Input}_\ell}$'])       








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
    meta = BBmeta(args.globals)

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
            if args.verbose: print("MPI TRUE, SIZE = ", size,", RANK = ", rank,'\n')

        except (ModuleNotFoundError, ImportError) as e:
            # Error handling
            print('ERROR IN MPI:', e)
            print('Proceeding without MPI\n')
            mpi = False
            rank=0
            pass

    tracemalloc.start()
    if args.verbose: print('Memory usage at the start is: ', tracemalloc.get_traced_memory())

    if args.sims:
        if args.verbose: print('Simulating maps ...')
        # meta_sims = BBmeta(args.sims)
  

        if not mpi:
            freq_maps_sim_list, fsky_binary, fg_freq_maps = wrapper_MakeSimsNOTMPI(args)

            # freq_maps_sim_list = []
            # for sim_num in range(meta_sims.general_pars['nsims']):

            #     if args.verbose: print('simulating maps sim number: ', sim_num + 1, '/', meta_sims.general_pars['nsims'], ' (for loop, NOT MPI)')
            #     if args.verbose: print('Memory usage before simulation is: ', tracemalloc.get_traced_memory())
                
            #     freq_maps, noise_maps, fg_freq_maps, cmb_sky, CMB_fg_freq_maps_beamed, fsky_binary = MakeSims(args)
                
            #     if args.verbose: print('Memory usage after simulation is: ', tracemalloc.get_traced_memory())
                
            #     freq_maps_sim_list.append(freq_maps)
                
            #     if args.verbose: print('saving map sims ...')
            #     np.save(os.path.join(meta_sims.comb_directory, 'comb_freq_maps_SIM'+str(sim_num).zfill(5)+'.npy' ),
            #              freq_maps )
            #     np.save(os.path.join(meta_sims.cmb_directory, 'cmb_maps_SIM'+str(sim_num).zfill(5)+'.npy' ), 
            #             cmb_sky )
            #     np.save(os.path.join(meta_sims.cmb_beamed_directory, 'cmb_beamed_freq_maps_SIM'+str(sim_num).zfill(5)+'.npy' ), 
            #             CMB_fg_freq_maps_beamed )
            #     np.save(os.path.join(meta_sims.fg_directory, 'fg_freq_maps_SIM'+str(sim_num).zfill(5)+'.npy' ), 
            #             fg_freq_maps )
            #     np.save(os.path.join(meta_sims.noise_directory, 'noise_freq_maps_SIM'+str(sim_num).zfill(5)+'.npy' ), 
            #             noise_maps )
        
        if mpi:
            freq_maps = wrapper_MakeSimsMPI(args, rank, size)

            # if meta_sims.general_pars['nsims'] != size:
            #     exit('ERROR: nsims must be equal to size in MPI mode. nsims = '+ str(meta_sims.general_pars['nsims'])+'  size = '+ str(size))

            # if args.verbose: print('simulating maps sim number: ', rank + 1, '/', meta_sims.general_pars['nsims'], ' (MPI)')
            # if args.verbose: print('Memory usage before simulation is: ', tracemalloc.get_traced_memory())
            # freq_maps, noise_maps, fg_freq_maps, cmb_sky, CMB_fg_freq_maps_beamed, fsky_binary = MakeSims(args)
            # if args.verbose: print('Memory usage after simulation is: ', tracemalloc.get_traced_memory())
            # np.save(os.path.join(meta_sims.comb_directory, 'comb_freq_maps_SIM'+str(rank).zfill(5)+'.npy' ), 
            #         freq_maps )
            # np.save(os.path.join(meta_sims.cmb_directory, 'cmb_maps_SIM'+str(rank).zfill(5)+'.npy' ), 
            #         cmb_sky )
            # np.save(os.path.join(meta_sims.cmb_beamed_directory, 'cmb_beamed_freq_maps_SIM'+str(rank).zfill(5)+'.npy' ),
            #          CMB_fg_freq_maps_beamed )
            # np.save(os.path.join(meta_sims.fg_directory, 'fg_freq_maps_SIM'+str(rank).zfill(5)+'.npy' ), 
            #         fg_freq_maps )
            # np.save(os.path.join(meta_sims.noise_directory, 'noise_freq_maps_SIM'+str(rank).zfill(5)+'.npy' ), 
            #         noise_maps )

        
    else:
        if args.verbose: print('Importing maps ...')
        # TODO: MPI
        # TODO: formating for importing real maps
        freq_maps = get_maps(args)

    
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
        


    if not mpi:
        wrapper_preprocNOTMPI(args, freq_maps_sim_list, fg_freq_maps, fsky_binary)

        # for sim_num in range(meta_sims.general_pars['nsims']):
        #     if args.verbose: print('Memory usage before pre-processing is: ', tracemalloc.get_traced_memory())
        #     if args.verbose: print('Pre-precessing freq-maps #', sim_num + 1, ' out of ', meta_sims.general_pars['nsims'])
        #     freq_maps_common_beamed = CommonBeamConvAndNsideModification(args, freq_maps_sim_list[sim_num])
        #     freq_maps_common_beamed_masked = ApplyBinaryMask(args, freq_maps_common_beamed)
        #     if args.verbose: print('Memory usage after pre-processing is: ', tracemalloc.get_traced_memory())
        #     if args.verbose: print('saving pre-processed maps ...')
        #     np.save(os.path.join(meta.pre_process_directory, 'freq_maps_common_beamed_masked'+str(sim_num).zfill(5)+'.npy' ),
        #             freq_maps_common_beamed_masked )
        #     np.save(os.path.join(meta.pre_process_directory, 'freq_maps_common_beamed'+str(sim_num).zfill(5)+'.npy' ),
        #             freq_maps_common_beamed )
        
        # if args.sims and args.plots:
        #     if args.verbose: print('checking pre-processed maps...\n')
        #     # This only checks the last simulations
        #     cl_preproc_freq_maps, model_beamed_total = check_preproc( args, freq_maps_common_beamed, fg_freq_maps, fsky_binary, sim_num=sim_num)
        


    if mpi:
        wrapper_preprocMPI(args, freq_maps, rank, size,  fg_freq_maps, fsky_binary)
        # if args.verbose: print('Pre-precessing freq-maps #', rank + 1, ' out of ', size, ' (MPI)')
        # if args.verbose: print('Memory usage before pre-processing is: ', tracemalloc.get_traced_memory())
        # freq_maps_common_beamed = CommonBeamConvAndNsideModification(args, freq_maps)
        # freq_maps_common_beamed_masked = ApplyBinaryMask(args, freq_maps_common_beamed)
        # if args.verbose: print('Memory usage after pre-processing is: ', tracemalloc.get_traced_memory())

        # if args.verbose: print('saving pre-processed maps ...\n')
        # np.save(os.path.join(meta.pre_process_directory, 'freq_maps_common_beamed_masked'+str(rank).zfill(5)+'.npy' ),
        #         freq_maps_common_beamed_masked )
        
        # if not meta_sims.noise_sim_pars['include_nhits']:
        #     # If nhits is used for the noise, all maps should be masked. 
        #     np.save(os.path.join(meta.pre_process_directory, 'freq_maps_common_beamed'+str(rank).zfill(5)+'.npy' ),
        #             freq_maps_common_beamed )
            
        # if args.sims and args.plots:
        #     if args.verbose: print('checking pre-processed maps...\n')
        #     if args.verbose: print('Memory usage before checking pre-processed maps is: ', tracemalloc.get_traced_memory())
        #     cl_preproc_freq_maps, model_beamed_total = check_preproc( args, freq_maps_common_beamed, fg_freq_maps, fsky_binary, sim_num=rank)        
        #     if args.verbose: print('Memory usage after checking pre-processed maps is: ', tracemalloc.get_traced_memory())
        #     # Ensure recvbuf is contiguous
        #     # to make sure the comm.Gather() works correctly

        #     cl_preproc_freq_maps = np.ascontiguousarray(cl_preproc_freq_maps) 

        #     recvbuf = None
        #     if rank == 0:
        #         shape_recvbuf = (size,) + cl_preproc_freq_maps.shape
        #         recvbuf = np.empty(shape_recvbuf)

        #         # Ensure recvbuf is contiguous
        #         # to make sure the comm.Gather() works correctly
        #         recvbuf = np.ascontiguousarray(recvbuf)

        #     comm.Gather(cl_preproc_freq_maps, recvbuf, root=0)

        #     if rank == 0:
        #         if args.verbose: print('checking MEAN preproc results...\n')

        #         mean_cl_preproc_freq_maps = np.mean(recvbuf, axis=0)
        #         plotTTEEBB_diff(args, mean_cl_preproc_freq_maps, model_beamed_total,
        #                         os.path.join( meta.plot_dir_from_output_dir(meta.pre_process_directory_rel), 'mean_preproc_cl_check.png'), 
        #                         legend_labels=[r'Mean preproc $C_\ell$ from map $\nu=$', r'Model Cl after preproc $\nu=$'], 
        #                         axis_labels=[r'$D_\ell \, [\mu K \, rad]^2$', r'$\frac{\Delta_{\ell}}{\rm{Input}_\ell}$'])       

    if args.verbose: print('Memory usage at the end is: ', tracemalloc.get_traced_memory())
    if rank == 0: 
        print('\n\nPre-Processing step completed succesfully\n\n')   

