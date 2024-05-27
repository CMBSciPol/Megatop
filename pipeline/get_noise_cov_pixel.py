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
        noise_cov_mean_ = noise_cov_mean[f]
        if mask_unseen is not None:
            noise_cov_mean_[np.where(mask_unseen==0)[0]] = hp.UNSEEN
        hp.mollview(noise_cov_mean_, cmap=cmap, cbar=True, hold=True, 
                    title=r'Noise cov map $\nu={}$ GHz'.format(meta.frequencies[f]),
                    norm=norm) 
        hp.graticule()

    map_noise_cov_save_path = os.path.join(plot_dir, file_name)
    plt.savefig(map_noise_cov_save_path)
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
    from pre_processing import CommonBeamConvAndNsideModification
    meta = BBmeta(args.globals)


    if args.sims:
        print('sims=True: Computing noise covariance from OnTheFlySims')
        meta_sims = BBmeta(args.sims)
        nsims = meta_sims.general_pars['nsims']

        noise_cov = np.zeros( [ len(meta.general_pars['frequencies']), 3,hp.nside2npix(meta_sims.map_sim_pars['nside_sim'])])
        noise_cov_preprocessed = np.zeros([ len(meta.general_pars['frequencies']), 3,hp.nside2npix(meta.general_pars['nside'])])

        for sim_num in range(nsims):
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
                                                    pixwin=False,fwhm=0.0,pol=True,verbose=False)  

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
                print('Applying nhits to both freq noise maps AND freq noise maps AFTER preprocessing.')
                nhits_map = hp.read_map(meta_sims._get_nhits_map_name())
                nhits_map_rescaled = nhits_map / max(nhits_map)
                freq_noise_maps /= np.sqrt(nhits_map_rescaled)
                freq_noise_maps[...,np.where(nhits_map_rescaled==0)[0]] = hp.UNSEEN

                # nhits_map_after_preproc = hp.read_map(meta._get_nhits_map_name())
                # nhits_map_after_preproc_rescaled = nhits_map_after_preproc / max(nhits_map_after_preproc)
                # freq_noise_maps_pre_processed /= np.sqrt(nhits_map_after_preproc_rescaled)

            freq_noise_maps_pre_processed = CommonBeamConvAndNsideModification(args, freq_noise_maps)


            noise_cov += freq_noise_maps**2
            noise_cov_preprocessed += freq_noise_maps_pre_processed**2
        
        noise_cov_mean = noise_cov/nsims
        noise_cov_preprocessed_mean = noise_cov_preprocessed/nsims


        if args.plots:
            add_test_param_in_save_name = ''
            if apply_pixwin:
                add_test_param_in_save_name +='_pixwinTrue'
           
            if not meta.noise_cov_pars['include_nhits']:
                add_test_param_in_save_name+='_nonhits'
                norm_maps = None
            else:
                norm_maps = 'hist'

                
            plot_cov_matrix(args, noise_cov_mean[:,0], 'map_noise_cov_T'+add_test_param_in_save_name+'.png', norm=norm_maps)
            plot_cov_matrix(args, noise_cov_mean[:,1], 'map_noise_cov_Q'+add_test_param_in_save_name+'.png', norm=norm_maps)
            plot_cov_matrix(args, noise_cov_mean[:,2], 'map_noise_cov_U'+add_test_param_in_save_name+'.png', norm=norm_maps)

            plot_cov_matrix(args, noise_cov_preprocessed_mean[:,0], 'map_noise_cov_preprocessed_T'+add_test_param_in_save_name+'.png', 
                            norm=norm_maps)
            plot_cov_matrix(args, noise_cov_preprocessed_mean[:,1], 'map_noise_cov_preprocessed_Q'+add_test_param_in_save_name+'.png', 
                            norm=norm_maps)
            plot_cov_matrix(args, noise_cov_preprocessed_mean[:,2], 'map_noise_cov_preprocessed_U'+add_test_param_in_save_name+'.png', 
                            norm=norm_maps)


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


def CheckNoiselvl(args, noise_cov, mask):
    meta = BBmeta(args.globals)

    fsky_binary = sum(mask) / len(mask) # only works if binary mask... 

    ell, N_ell_P_SA, Map_white_noise_levels = V3.so_V3_SA_noise(
    sensitivity_mode = meta.general_pars['sensitivity_mode'],
    one_over_f_mode = 2, # fixed to None since we only use white noise here
    SAC_yrs_LF = meta.general_pars['SAC_yrs_LF'], f_sky = fsky_binary, 
    ell_max = meta.general_pars['lmax'], delta_ell=1,
    beam_corrected=False, remove_kluge=False, CMBS4='')

    std_map = np.std(noise_cov, axis = -1)
    std_uK_arcmin = std_map * hp.nside2resol(hp.npix2nside(noise_cov.shape[-1]))

    print('std_uK_arcmin = ', std_uK_arcmin)
    print('std_uK_arcmin / Map_white_noise_levels = ', std_uK_arcmin[:,1] / Map_white_noise_levels)
    return std_uK_arcmin



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='simplistic simulator')
    parser.add_argument("--globals", type=str,
                        help="Path to yaml with global parameters")
    parser.add_argument("--sims", default=None,
                        help="Generate a set of sims if True.")    
    parser.add_argument("--plots", action="store_true",
                        help="Plot the generated maps if True.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    noise_cov, noise_cov_preprocessed = GetNoiseCov(args)

    testing_pixwin = True
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

    np.save(os.path.join(meta.output_dirs['root'], meta.output_dirs['covmat_directory'], 'pixel_noise_cov.npy' ),
                noise_cov )
    np.save(os.path.join(meta.output_dirs['root'], meta.output_dirs['covmat_directory'], 'pixel_noise_cov_preprocessed.npy' ),
                noise_cov_preprocessed )    
    IPython.embed()