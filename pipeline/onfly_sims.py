import argparse
from megatop.metadata_manager import BBmeta
import warnings
import healpy as hp
import numpy as np
from fgbuster.observation_helpers import get_instrument, get_sky, get_observation, standardize_instrument
import os
import matplotlib.pyplot as plt
import megatop.V3calc as V3
from megatop.utils import get_Cl_CMB_model_from_meta


SO_FREQS = [27, 39, 93, 145, 220, 280] #TODO: Dodgy ...

#TODO use logger instead of prints

def make_sims(meta, verbose=True, plots=False, noise_only=False):
    """This routine creates mock data from the map sets.

    Args:
        meta: metadata_object, initialised from the config file. #TODO to complete

    Returns:
        CMB_fg_noise_freq_maps (ndarray): The frequency maps of the CMB, foregrounds, and noise, with shape (num_freq, num_stokes, num_pixels).
        
        noise_maps (ndarray): The noise frequency maps, with shape (num_freq, num_stokes, num_pixels).
        
        fg_freq_maps (ndarray): The foreground frequency maps, with shape (num_freq, num_stokes, num_pixels).
        
        cmb_sky (ndarray): The CMB sky map, with shape (num_stokes, num_pixels).
        
        CMB_fg_freq_maps_beamed (ndarray): The frequency maps of the CMB and foregrounds after beam convolution, with shape (num_freq, num_stokes, num_pixels). TODO: Needed ?!?!
        
        fsky_binary (float): The fraction of sky covered by the binary mask.

    """
    meta.timer.start('sim')
    nside = meta.general_pars["nside"]

    #Initializing instrument
    binary_mask = meta.read_mask('binary')        
    fsky_binary = sum(binary_mask) / len(binary_mask)

    n_ell, map_white_noise_levels, beams_FWHM = get_noise_beams(meta, fsky_binary=fsky_binary)

    instrument_config = {
        'frequency' : meta.frequencies,
        'depth_i' : map_white_noise_levels/np.sqrt(2),
        'depth_p' : map_white_noise_levels
        }

    instrument = standardize_instrument(instrument_config) 

      #Creating noise maps 
    meta.timer.start('noise')

    if meta.noise_sim_pars['noise_option']=='white_noise':
        if verbose: print('White noise only case')
        # nlev_map = cmb_fg_freq_maps_beamed*0.0
        nlev_map = np.zeros((len(meta.frequencies), 3, hp.nside2npix(nside)))
        for i_f,f in enumerate(instrument.frequency): 
            nlev_map[i_f] = np.array([instrument.depth_i[i_f], instrument.depth_p[i_f], instrument.depth_p[i_f]])[:,np.newaxis]*np.ones((3,hp.nside2npix(nside)))
        nlev_map /= hp.nside2resol(nside, arcmin=True)
        noise_maps = np.random.normal( np.zeros((len(meta.frequencies), 3, hp.nside2npix(nside))), 
                                      nlev_map, (len(meta.frequencies), 3, hp.nside2npix(nside)))
    elif meta.noise_sim_pars['noise_option']=='no_noise':
        if verbose: print('No noise case')
        noise_maps = np.zeros((len(meta.frequencies), 3, hp.nside2npix(nside)))
    elif meta.noise_sim_pars['noise_option']=='noise_spectra':
        if verbose: print('Full noise spectra case')
        print("NOT TESTED YET !!!!")#TODO TEST THIS !!!!
        noise_maps = np.zeros((len(meta.frequencies), 3, hp.nside2npix(nside)))
        for i_f, f in enumerate(meta.frequencies):
            noise_maps[i_f] = hp.synfast(3*(n_ell[i_f],)+3*(None,), #using same noise spectra for T, E and B !
                                         new=True, 
                                         pixwin=False, 
                                         nside = nside) 
    else:
        raise ValueError('ERROR: Other noise cases not handled yet...')
 
    if meta.noise_sim_pars['noise_option'] != 'no_noise' and meta.noise_sim_pars['include_nhits']:
        if verbose: print('Including nhits in noise maps...')
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
    
    meta.timer.stop('noise', "Adding noise", verbose)

    if noise_only:
        if verbose: print("Simulating noise only maps")
        return None, None, noise_maps

    # Performing the CMB simulation with synfast
    meta.timer.start('cmb')
    if meta.map_sim_pars['cmb_sim_no_pysm']:
        if plots:
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
        Cl_cmb_model = get_Cl_CMB_model_from_meta(meta)
        
        if meta.map_sim_pars['fixed_cmb']:
            # Fixing seed so that the CMB is the same for all sims.
            # WARNING: highly wasteful as it will generate the same CMB for all sims and store them all
            # TODO: Optimize!
            np.random.seed(0) 
        cmb_map = hp.synfast(Cl_cmb_model[0], #[Cl_TT, Cl_EE, Cl_BB, Cl_TE, Cl_EE*0.0, Cl_EE*0.0], 
                             nside=nside, new=True, pixwin=False)
        if meta.map_sim_pars['fixed_cmb']:
            np.random.seed(None) # Resetting seed after CMB generation
    meta.timer.stop('cmb', 'CMB simulation', verbose)

    # Creating Pysm Fg maps 
    meta.timer.start('fg')
    tag=''
    for s in meta.sky_model:
        tag+=s
    sky = get_sky(nside, tag=tag)

    fg_freq_maps = get_observation(instrument, sky, noise=False) 
    cmb_fg_freq_maps = fg_freq_maps + cmb_map

    if verbose: print('Beaming sky maps...', end=" ")
    cmb_fg_freq_maps_beamed = []

    for i_f,f in enumerate(meta.frequencies):

        if verbose: print(f, end=" ")
        lmax_convolution = 3* nside # here lmax seems to play an important role            
        cmb_fg_alms_T, cmb_fg_alms_Q, cmb_fg_alms_U = hp.map2alm(cmb_fg_freq_maps[i_f], lmax=lmax_convolution, pol=True)
        Bl_gauss_fwhm = hp.gauss_beam( np.radians(beams_FWHM[i_f]/60), lmax=lmax_convolution, pol=True)
        
        wpix_in = hp.pixwin( nside, pol=True, lmax=lmax_convolution) # Pixel window function of input maps
          
        sm_corr_T =  Bl_gauss_fwhm[:,0] * wpix_in[0]
        sm_corr_P =  Bl_gauss_fwhm[:,1] * wpix_in[1]

        #change beam and wpix
        alm_out_T = hp.almxfl(cmb_fg_alms_T, sm_corr_T)
        alm_out_E = hp.almxfl(cmb_fg_alms_Q, sm_corr_P)
        alm_out_B = hp.almxfl(cmb_fg_alms_U, sm_corr_P)

        #alm-->mapf
        cmb_fg_alms_out_T, cmb_fg_alms_out_Q, cmb_fg_alms_out_U = hp.alm2map([alm_out_T,alm_out_E,alm_out_B], nside,
                                                    lmax=lmax_convolution, pixwin=False, fwhm=0.0, pol=True) 


        cmb_fg_freq_maps_beamed.append([cmb_fg_alms_out_T, cmb_fg_alms_out_Q, cmb_fg_alms_out_U])        
    cmb_fg_freq_maps_beamed = np.array(cmb_fg_freq_maps_beamed)
    if verbose: print("")
    meta.timer.stop('fg', "Generating foreground maps", verbose)
    
    cmb_fg_noise_freq_maps = cmb_fg_freq_maps_beamed + noise_maps

    if meta.noise_sim_pars['include_nhits']:  #TODO: This is not used for now !!
        # Importing mask from the simulations nside to apply to simulated maps
        # Necessary to do it here when applying 1/sqrt(nhits) to the noise maps as it will create inf in the noise maps
        binary_mask_sim = meta.read_mask('binary')
        cmb_fg_noise_freq_maps[...,np.where(binary_mask_sim==0)[0]] = 0 # hp.UNSEEN

    meta.timer.stop('sim', "Simulating one sky", verbose)

    return cmb_fg_noise_freq_maps, cmb_fg_freq_maps_beamed, noise_maps #TODO check if cmb_fg_noise_freq_maps is needed or not ...


def get_noise_beams(meta, fsky_binary, verbose=True):
    '''
    This function returns the noise and beams depending on the noise_sim_pars settings in the config file.

    Args:
        meta: metadata_manager object containing all the config file options
        fsky_binary (float): The fraction of sky covered by the binary mask.
        
    Returns:
        n_ell (ndarray or None): the noise spectra (if computed) shape is (num_freq, num_ell)
        map_white_noise_level (list): the (polarisation) white noise levels, shape is (num_freqs)
        beams (list): the beams FWHM (in arcmin), shape is (num_freqs)
    '''
    if meta.noise_sim_pars['experiment'] == 'SO':
        idx_freqs = meta.idx_from_list(SO_FREQS)
        _, n_ell, map_white_noise_levels = V3.so_V3_SA_noise(
                sensitivity_mode = meta.noise_sim_pars['sensitivity_mode'],
                one_over_f_mode = meta.noise_sim_pars['knee_mode'], 
                SAC_yrs_LF = meta.noise_sim_pars['SAC_yrs_LF'], f_sky = fsky_binary, 
                ell_max = meta.general_pars['lmax'], delta_ell=1,
                beam_corrected=False, remove_kluge=False, CMBS4='')
        beams = V3.so_V3_SA_beams()
        beams = beams[idx_freqs]
        map_white_noise_levels =  map_white_noise_levels[idx_freqs]
        if verbose: print('Map_white_noise_levels = ', map_white_noise_levels)
    else: 
        print("ONLY SO IMPLEMETED FOR NOW") #TODO implement custom experiment?
    return n_ell, map_white_noise_levels, beams

def get_NL_from_white_noise(meta, map_white_noise_levels):
    lmax_convolution = 3*meta.general_pars['nside']

    N_ell_white_f_arcmin = []
    for i_f in range(len(meta.frequencies)):
        N_ell_white_f_arcmin.append(np.ones(lmax_convolution) *
                            map_white_noise_levels[i_f]**2)
    N_ell_white_f = np.array(N_ell_white_f_arcmin)  * (np.pi/60/180)**2 
    N_ell_white_f_temp = np.array(N_ell_white_f_arcmin)  * (np.pi/60/180)**2 /2

    n_ell = np.empty([len(meta.frequencies), 3, lmax_convolution])
    n_ell[:,0] = N_ell_white_f_temp
    n_ell[:,1] = N_ell_white_f
    n_ell[:,2] = N_ell_white_f  

    return n_ell

def plotTTEEBB_diff(meta, Cl_data, Cl_model, save_name, 
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

def check_sims(meta, cmb_fg_freq_maps_beamed):
        '''	    
        This function checks the simulated maps by comparing their Cls with the input CMB spectra beamed, the foreground Cls computed by their input map beamed
        and the white noise Cl. It saves the different plots in the plots directory of the simulation output directory.
        
        Args:
            meta: The parser arguments, containing the path to the global parameters file to set up metadata manager.
            cmb_sky (ndarray): The CMB sky map, with shape (num_stokes, num_pixels).
            noise_maps (ndarray): The noise maps, with shape (num_freq, num_stokes, num_pixels).
            freq_maps (ndarray): The frequency maps, with shape (num_freq, num_stokes, num_pixels).
            fg_freq_maps (ndarray): The foreground frequency maps, with shape (num_freq, num_stokes, num_pixels).
            CMB_fg_freq_maps_beamed (ndarray): The frequency maps of the CMB and foregrounds after beam convolution, with shape (num_freq, num_stokes, num_pixels).
            fsky_binary (float): The fraction of sky covered by the binary mask.
        
        Returns:
            None
        '''
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
        
def save_sims(meta,cmb_fg_freq_maps_beamed):
    for i_f, map in enumerate(meta.maps_list):
        fname=os.path.join(meta.map_directory, meta.map_sets[map]['file_root']+'.fits')
        hp.write_map(fname, cmb_fg_freq_maps_beamed[i_f], dtype=['float64', 'float64', 'float64'], overwrite=True)

def save_noise_maps(meta, noise_maps, id=0):
    fname=os.path.join(meta.mock_directory, 'noise_sim_id_{:04d}'.format(id))
    np.save(fname, noise_maps)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='simplistic simulator')
    parser.add_argument("--globals", type=str,
                        help="Path to yaml with global parameters") 
    parser.add_argument("--plots", action="store_true",
                        help="Plot various outputs maps if True.")
    parser.add_argument("--sim_id", type=int,
                        help="Id of the simulation (useful fo noise covariance estimation)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    meta = BBmeta(args.globals)
    cmb_fg_noise_freq_maps, cmb_fg_freq_maps_beamed, noise_maps = make_sims(meta, verbose=args.verbose, plots=args.plots)
    check_sims(meta, cmb_fg_freq_maps_beamed)
    if meta.noise_sim_pars["save_noise_sim"]:
        if args.sim_id is not None:
            save_noise_maps(meta,noise_maps,args.sim_id)
        else:
            save_noise_maps(meta,noise_maps)

    save_sims(meta, cmb_fg_freq_maps_beamed)