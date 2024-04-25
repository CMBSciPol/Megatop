import argparse
from megatop import BBmeta, utils
import numpy as np
import os
import healpy as hp
import matplotlib.pyplot as plt
import glob


import sys
sys.path.append('/global/cfs/cdirs/sobs/users/krach/BBSims/NOISE_20201207/')
from combine_noise import compute_noise_factors, combine_noise_maps

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




def get_cov_matrix_pixel(args):
    """
    """
    meta = BBmeta(args.globals)


    if meta.config['external_noise_cov']:
        print('/// EXT NOISE COV')
        noise_cov = hp.read_map(meta.config['external_noise_cov'], field=None)
        noise_cov_beamed = noise_cov*1.0
    elif meta.config['bypass_noise_cov']:
        print('/// BYPASS NOISE COV')
        tag = ''
        for f in meta.frequency: tag += str(f)+'_'
        for key in ['common_beam_correction', 'no_inh', 'Nico_noise_combination', 'Nsims_bias', 'nside', 'sensitivity_mode', 'knee_mode']:
            tag += key+'_'+str(meta.config[key])
        path_to_noise_cov = os.path.join('/global/cscratch1/sd/josquin/bypass_noise_cov_'+tag)
        if not os.path.exists(path_to_noise_cov+'.npy'):
            print('noise covariance is not on disk yet. Computing it now.')
            noise_cov, noise_cov_beamed = noise_covariance_estimation(meta, freq_maps.shape, instrument, nhits)
            np.save(path_to_noise_cov, (noise_cov, noise_cov_beamed), allow_pickle=True)
        else: 
            noise_cov, noise_cov_beamed = np.load(path_to_noise_cov+'.npy')
    else:
        print('/// WHITE NOISE COV')
        noise_cov = freq_maps*0.0
        # nlev /= hp.nside2resol(meta.config['nside'], arcmin=True)
        noise_cov[::3,:] = nlev[:,np.newaxis]/np.sqrt(2.0)
        noise_cov[1::3,:] = nlev[:,np.newaxis]
        noise_cov[2::3,:] = nlev[:,np.newaxis]
        noise_cov *= binary_mask
        # divind by the pixel size in arcmin
        noise_cov /=  hp.nside2resol(meta.config['nside'], arcmin=True)
        if meta.config['noise_option']!='white_noise' and meta.config['noise_option']!='no_noise':
            noise_cov /= np.sqrt(nhits/np.amax(nhits))
        # we put it to square !
        noise_cov *= noise_cov
        noise_cov_beamed = noise_cov*1.0

    if ((meta.config['common_beam_correction']!=0.0) and (not meta.config['bypass_noise_cov'])):
        print('/////////// noise_cov_beam_correction after beam convolution ///////////////')
        noise_cov_beamed = noise_covariance_correction(cov_in=noise_cov, instrument=instrument_config, 
                        common_beam=meta.config['common_beam_correction'], nside_in=NSIDE_INPUT_MAP, 
                            nside_out=meta.config['nside'], Nsims=meta.config['Nsims_bias'])


    noise_cov[:,np.where(binary_mask==0)[0]] = hp.UNSEEN
    noise_cov_beamed[:,np.where(binary_mask==0)[0]] = hp.UNSEEN



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='simplistic simulator')
    parser.add_argument("--globals", type=str,
                        help="Path to yaml with global parameters")
    parser.add_argument("--sims", action="store_true",
                        help="Generate a set of sims if True.")
    parser.add_argument("--plots", action="store_true",
                        help="Plot the generated maps if True.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.sims and args.plots:
        warnings.warn("Both --sims and --plot are set to True. "
                      "Too many plots will be generated. "
                      "Set --plot to False")
        args.plots = False
    get_cov_matrix_pixel(args)    
