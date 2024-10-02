import argparse
from megatop.metadata_manager import BBmeta, Timer
import warnings
import healpy as hp
import numpy as np
from fgbuster.observation_helpers import get_instrument, get_sky, get_observation, standardize_instrument
import os
import matplotlib.pyplot as plt
import megatop.V3calc as V3


SO_FREQS = [27, 39, 93, 145, 220, 280]


#TODO use logger instead of prints

def make_sims(args):
    timer_sims = Timer()
    timer_sims.start('sim')
    """This routine creates mock data from the map sets.

    Args:
        args: The parser arguments, containing the path to the global parameters file and simulation file
              to set up metadata managers.

    Returns:
        CMB_fg_noise_freq_maps (ndarray): The frequency maps of the CMB, foregrounds, and noise, with shape (num_freq, num_stokes, num_pixels).
        
        noise_maps (ndarray): The noise frequency maps, with shape (num_freq, num_stokes, num_pixels).
        
        fg_freq_maps (ndarray): The foreground frequency maps, with shape (num_freq, num_stokes, num_pixels).
        
        cmb_sky (ndarray): The CMB sky map, with shape (num_stokes, num_pixels).
        
        CMB_fg_freq_maps_beamed (ndarray): The frequency maps of the CMB and foregrounds after beam convolution, with shape (num_freq, num_stokes, num_pixels). TODO: Needed ?!?!
        
        fsky_binary (float): The fraction of sky covered by the binary mask.

    """
    meta = BBmeta(args.globals)

    d_config = meta.map_sim_pars['dust_model']
    s_config = meta.map_sim_pars['sync_model']

    # Performing the CMB simulation with synfast
    timer_sims.start('cmb')
    if meta.map_sim_pars['cmb_sim_no_pysm']:
        if args.plots:
            path_Cl_BB_lens = meta.get_fname_cls_fiducial_cmb('lensed')
            path_Cl_BB_prim_r1 = meta.get_fname_cls_fiducial_cmb('unlensed_scalar_tensor_r1')

            Cl_lens = hp.read_cl(path_Cl_BB_lens)
            Cl_prim = hp.read_cl(path_Cl_BB_prim_r1)[...,:Cl_lens.shape[-1]]
            Cl_BB_prim = meta.map_sim_pars['r_input']*Cl_prim[2]

            ell_range = np.arange(Cl_lens.shape[-1])
            todls = ell_range*(ell_range+1)/2./np.pi
            
            plt.plot(ell_range, todls * Cl_prim[0], label='prim TT', color='C0')
            plt.plot(ell_range, todls * Cl_lens[0], label='lens TT', color='C0', ls='--')

            plt.plot(ell_range, todls * Cl_prim[1], label='prim EE', color='C1')
            plt.plot(ell_range, todls * Cl_lens[1], label='lens EE', color='C1', ls='--')

            plt.plot(ell_range, todls * Cl_BB_prim, label='prim BB', color='C2')
            plt.plot(ell_range, todls * Cl_lens[2], label='lens BB', color='C2', ls='--')

            plt.plot(ell_range, todls * Cl_prim[3], label='prim TE', color='C3')
            plt.plot(ell_range, todls * Cl_lens[3], label='lens TE', color='C3', ls='--')

            plt.loglog()
            plt.xlabel(r'$\ell$')
            plt.ylabel(r'$D_\ell$')
            plt.legend()
            plt.savefig(os.path.join(meta.plots_directory, 'fiducial_CMB_spectra.png'), bbox_inches='tight') 
            plt.close()
        Cl_cmb_model = get_Cl_CMB_model_from_meta(args)
        
        if meta.map_sim_pars['fixed_cmb']:
            # Fixing seed so that the CMB is the same for all sims.
            # WARNING: highly wasteful as it will generate the same CMB for all sims and store them all
            # TODO: Optimize!
            np.random.seed(0) 
        cmb_map = hp.synfast(Cl_cmb_model[0], #[Cl_TT, Cl_EE, Cl_BB, Cl_TE, Cl_EE*0.0, Cl_EE*0.0], 
                             nside=meta.map_sim_pars['nside_sim'], new=True, pixwin=False)
        if meta.map_sim_pars['fixed_cmb']:
            np.random.seed(None) # Resetting seed after CMB generation
    else:
        raise ValueError('CMB sims only handled using synfast on fiducial Cls. TODO: get CMB from pysm ?') #TODO
    timer_sims.stop('cmb', 'CMB simulation', args.verbose)

    #Initializing instrument
    binary_mask = meta.read_mask('binary')        
    fsky_binary = sum(binary_mask) / len(binary_mask)

    map_white_noise_levels = get_SO_white_noise(args, fsky_binary=fsky_binary)

    instrument_config = {
        'frequency' : meta.frequencies,
        'depth_i' : map_white_noise_levels/np.sqrt(2),
        'depth_p' : map_white_noise_levels
        }

    instrument = standardize_instrument(instrument_config)

    # Creating Pysm Fg maps 
    timer_sims.start('fg')
    sky = get_sky(meta.map_sim_pars['nside_sim'], d_config+s_config)

    fg_freq_maps = get_observation(instrument, sky, noise=False) 
    cmb_fg_freq_maps = fg_freq_maps + cmb_map

    if args.verbose: print('Beaming sky maps...', end=" ")
    cmb_fg_freq_maps_beamed = []

    for i_f,f in enumerate(meta.frequencies):

        if args.verbose: print(f, end=" ")

        lmax_convolution = 3* meta.general_pars['nside']

        # here lmax seems to play an important role            
        cmb_fg_alms_T, cmb_fg_alms_Q, cmb_fg_alms_U = hp.map2alm(cmb_fg_freq_maps[i_f], lmax=lmax_convolution, pol=True)

        Bl_gauss_fwhm = hp.gauss_beam( np.radians(meta.pre_proc_pars['fwhm'][i_f]/60), lmax=lmax_convolution, pol=True)
        
        wpix_in = hp.pixwin( meta.general_pars['nside'], pol=True, lmax=lmax_convolution) # Pixel window function of input maps
          
        sm_corr_T =  Bl_gauss_fwhm[:,0] * wpix_in[0]
        sm_corr_P =  Bl_gauss_fwhm[:,1] * wpix_in[1]

        #change beam and wpix
        alm_out_T = hp.almxfl(cmb_fg_alms_T, sm_corr_T)
        alm_out_E = hp.almxfl(cmb_fg_alms_Q, sm_corr_P)
        alm_out_B = hp.almxfl(cmb_fg_alms_U, sm_corr_P)

        #alm-->mapf
        cmb_fg_alms_out_T, cmb_fg_alms_out_Q, cmb_fg_alms_out_U = hp.alm2map([alm_out_T,alm_out_E,alm_out_B], meta.general_pars['nside'],
                                                    lmax=lmax_convolution, pixwin=False, fwhm=0.0, pol=True) 


        cmb_fg_freq_maps_beamed.append([cmb_fg_alms_out_T, cmb_fg_alms_out_Q, cmb_fg_alms_out_U])        
    cmb_fg_freq_maps_beamed = np.array(cmb_fg_freq_maps_beamed)
    if args.verbose: print("")
    timer_sims.stop('fg', "Generating foreground maps", args.verbose)
    
    #Creating noise maps...
    timer_sims.start('noise')

    if meta.noise_sim_pars['noise_option']=='white_noise':
        nlev_map = cmb_fg_freq_maps_beamed*0.0
        for i_f,f in enumerate(instrument.frequency): 
            nlev_map[i_f] = np.array([instrument.depth_i[i_f], instrument.depth_p[i_f], instrument.depth_p[i_f]])[:,np.newaxis]*np.ones((3,cmb_fg_freq_maps_beamed.shape[-1]))
        nlev_map /= hp.nside2resol(meta.map_sim_pars['nside_sim'], arcmin=True)
        noise_maps = np.random.normal(cmb_fg_freq_maps_beamed*0.0, nlev_map, cmb_fg_freq_maps_beamed.shape)
    elif meta.noise_sim_pars['noise_option']=='':
        if args.verbose: print('No noise case')
        noise_maps = 0 * fg_freq_maps
    else:
        raise ValueError('ERROR: Other noise cases not handled yet...')
 
    if meta.noise_sim_pars['noise_option'] != '' and meta.noise_sim_pars['include_nhits']:
        if args.verbose: print('Including nhits in noise maps...')
        nhits_map = meta.read_hitmap() 
        nhits_map_rescaled = nhits_map / max(nhits_map)
        binary_mask_sim = meta.read_mask('binary')

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

    cmb_fg_noise_freq_maps = cmb_fg_freq_maps_beamed + noise_maps

    if meta.noise_sim_pars['include_nhits']:
        # Importing mask from the simulations nside to apply to simulated maps
        # Necessary to do it here when applying 1/sqrt(nhits) to the noise maps as it will create inf in the noise maps
        binary_mask_sim = meta.read_mask('binary')
        cmb_fg_noise_freq_maps[...,np.where(binary_mask_sim==0)[0]] = 0 # hp.UNSEEN
    timer_sims.stop('noise', "Adding noise", args.verbose)
    timer_sims.stop('sim', "Simulating one sky", args.verbose)

    return cmb_fg_noise_freq_maps, cmb_fg_freq_maps_beamed, noise_maps, fsky_binary

def get_Cl_CMB_model_from_meta(args):
    '''
    This function reads the fiducial CMB Cls from the metadata manager and combines scalar, lensing and tensor 
    contributions to return the model Cls according to A_lens and r in the simulation parameter file.

    Args:
        args: The parser arguments, containing the path to the global parameters file to set up metadata manager.
        
    Returns:
        Cl_cmb_model (ndarray): The model CMB Cls, with shape (num_freq, num_spectra [TT,EE,BB,TE,EB,TB], num_ell).
    '''
    meta = BBmeta(args.globals)
    path_Cl_BB_lens = meta.get_fname_cls_fiducial_cmb('lensed')
    path_Cl_BB_prim_r1 = meta.get_fname_cls_fiducial_cmb('unlensed_scalar_tensor_r1')

    Cl_BB_prim = meta.map_sim_pars['r_input']*hp.read_cl(path_Cl_BB_prim_r1)[2]
    Cl_lens = hp.read_cl(path_Cl_BB_lens)

    l_max_lens = len(Cl_lens[0])
    Cl_BB_lens = meta.map_sim_pars['A_lens']*Cl_lens[2]
    Cl_TT = Cl_lens[0]
    Cl_EE = Cl_lens[1]
    Cl_TE = Cl_lens[3]

    Cl_BB = Cl_BB_prim[:l_max_lens] + Cl_BB_lens
    Cl_cmb_model = np.array([[Cl_TT, Cl_EE, Cl_BB, Cl_TE, Cl_EE*0.0, Cl_EE*0.0]])    
    return Cl_cmb_model

def get_SO_white_noise(args, fsky_binary):
    '''
    This function computes the white noise level from V3calc using parameters from the metadata manager 
    and returns the noise Cls.

    Args:
        args: The parser arguments, containing the path to the global parameters file to set up metadata manager.
        fsky_binary (float): The fraction of sky covered by the binary mask.
        
    Returns:
        model_noise (ndarray): The model noise Cls, with shape (num_freq, num_spectra [TT,EE,BB], num_ell).
    '''
    meta = BBmeta(args.globals)
    idx_freqs = meta.idx_from_list(SO_FREQS)

    _, _, map_white_noise_levels = V3.so_V3_SA_noise(
            sensitivity_mode = meta.noise_sim_pars['sensitivity_mode'],
            one_over_f_mode = 2, # fixed to None since we only use white noise here
            SAC_yrs_LF = meta.noise_sim_pars['SAC_yrs_LF'], f_sky = fsky_binary, 
            ell_max = meta.general_pars['lmax'], delta_ell=1,
            beam_corrected=False, remove_kluge=False, CMBS4='' )
    
    map_white_noise_levels =  map_white_noise_levels[idx_freqs]
    if args.verbose: print('Map_white_noise_levels = ', map_white_noise_levels)

    return map_white_noise_levels

def get_NL_from_white_noise(args, map_white_noise_levels):
    meta = BBmeta(args.globals)
    lmax_convolution = 3*meta.general_pars['nside']

    N_ell_white_f_arcmin = []
    for i_f in range(len(meta.frequencies)):
        N_ell_white_f_arcmin.append(np.ones(lmax_convolution) *
                            map_white_noise_levels[i_f]**2)
    N_ell_white_f = np.array(N_ell_white_f_arcmin)  * (np.pi/60/180)**2 
    N_ell_white_f_temp = np.array(N_ell_white_f_arcmin)  * (np.pi/60/180)**2 /2

    model_noise = np.empty([len(meta.frequencies), 3, lmax_convolution])
    model_noise[:,0] = N_ell_white_f_temp
    model_noise[:,1] = N_ell_white_f
    model_noise[:,2] = N_ell_white_f  

    return model_noise

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
        plt.savefig(os.path.join(meta.plots_directory, save_name), bbox_inches='tight') 
        plt.close()

def check_sims(args, cmb_fg_freq_maps_beamed):
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

        meta = BBmeta(args.globals)
        print("NO CHECKS DONE FOR NOW !!!!!") #TODO
        # # cl_cmb_sky_maps = hp.anafast(cmb_sky)

        # cl_noise_f = []
        # cl_cmb_fg_freq_maps_beamed = []

        # for i_f in range(len(meta.frequencies)):
        #     cl_noise_f.append( hp.anafast(noise_maps[i_f]) )
        #     cl_cmb_fg_freq_maps_beamed.append(hp.anafast(cmb_fg_freq_maps_beamed[i_f]))

        # cl_noise_f = np.array(cl_noise_f)
        # cl_cmb_fg_freq_maps_beamed = np.array(cl_cmb_fg_freq_maps_beamed)

        # map_white_noise_levels = get_SO_white_noise(args, fsky_binary)
        # model_noise = get_NL_from_white_noise(args, map_white_noise_levels)

        # plotTTEEBB_diff(args, cl_noise_f * fsky_binary, model_noise, 
        #                 os.path.join(meta.plots_directory, 'Noise_lvl_check.png'), 
        #                 legend_labels=[r'Noise $C_\ell$ from map $\nu=$', r'Input white noise lvl $\nu=$'], 
        #                 axis_labels=[r'$D_\ell \, [\mu K \, rad]^2$', r'$\frac{\Delta_{\ell}}{\rm{Input}_\ell}$'])

        # Cl_cmb_model = get_Cl_CMB_model_from_meta(args)

        # Bl_gauss_fwhm_freq = []
        # for f in range(len(meta.general_pars['frequencies'])):
        #     Bl_gauss_fwhm_freq.append( hp.gauss_beam( np.radians(meta.pre_proc_pars['fwhm'][f]/60), lmax=3*meta.map_sim_pars['nside_sim']-1, pol=True))
        # Bl_gauss_fwhm_freq = np.array(Bl_gauss_fwhm_freq)

        # wpix_in = hp.pixwin( meta.general_pars['nside'], pol=True, lmax=3*meta.map_sim_pars['nside_sim']-1) # Pixel window function of input maps


        # #TODO : Need the cl_fg_freq_maps !!!
        # beamed_sky_TT = (Cl_cmb_model[...,:3*meta.map_sim_pars['nside_sim']] + cl_fg_freq_maps)[:,0] * Bl_gauss_fwhm_freq[...,0]**2 * wpix_in[0]**2
        # beamed_sky_EE = (Cl_cmb_model[...,:3*meta.map_sim_pars['nside_sim']] + cl_fg_freq_maps)[:,1] * Bl_gauss_fwhm_freq[...,1]**2 * wpix_in[1]**2
        # beamed_sky_BB = (Cl_cmb_model[...,:3*meta.map_sim_pars['nside_sim']] + cl_fg_freq_maps)[:,2] * Bl_gauss_fwhm_freq[...,1]**2 * wpix_in[1]**2

        # beamed_sky_model = np.array([ beamed_sky_TT, beamed_sky_EE, beamed_sky_BB ]).swapaxes(0,1)

        # plotTTEEBB_diff(args, cl_cmb_fg_freq_maps_beamed * fsky_binary, beamed_sky_model * fsky_binary, 
        #                 os.path.join(meta.plots_directory, 'sky_beamed_check.png') , 
        #                 legend_labels=[r'$C_{\ell}^{\rm CMB+fg beamed}$ from map', r'Input $C_{\ell}^{\rm CMB+fg beamed}$'], 
        #                 axis_labels=[r'$D_\ell \, [\mu K \, rad]^2$', r'$\frac{\Delta_{\ell}}{\rm{Input}_\ell}$'])
        
def save_sims(args,cmb_fg_freq_maps_beamed):
    meta = BBmeta(args.globals)
    for i, map in enumerate(meta.maps_list):
        fname=os.path.join(meta.map_directory, meta.map_sets[map]['file_root']+'.fits')
        hp.write_map(fname, cmb_fg_freq_maps_beamed[i], dtype=['float64', 'float64', 'float64'], overwrite=True)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='simplistic simulator')
    parser.add_argument("--globals", type=str,
                        help="Path to yaml with global parameters")
    parser.add_argument("--use_mpi", action="store_true",
                        help="Use MPI instead of for loops to pre-process multiple maps, or simulate multiple sims.")
    parser.add_argument("--sims", action="store_true",
                        help="Generate a set of sims if True.")    
    parser.add_argument("--plots", action="store_true",
                        help="Plot the generated maps if True.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    meta = BBmeta(args.globals)
    if args.sims and not (hasattr(meta,'map_sim_pars') and hasattr(meta, 'noise_sim_pars')):
        print(f"Error with config file at {args.globals} !!")
        raise Exception("Missing map_sim_pars or noise_sim_pars fields ")
    else:
        cmb_fg_noise_freq_maps, cmb_fg_freq_maps_beamed, noise_maps, fsky_binary = make_sims(args)
        check_sims(args, cmb_fg_freq_maps_beamed)
        save_sims(args, cmb_fg_freq_maps_beamed)