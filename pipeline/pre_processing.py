import argparse
import IPython
from metadata_manager import BBmeta
import warnings
import healpy as hp
import numpy as np
from fgbuster.observation_helpers import get_instrument, get_sky, get_observation, standardize_instrument
import glob
import os
import sys
sys.path.append('/global/cfs/cdirs/sobs/users/krach/BBSims/NOISE_20201207/')
from combine_noise import *

def get_combined_map_sims(args):
    meta = BBmeta(args.globals)
    # TODO: the line below might no be necessary anymore
    if not os.path.exists(meta.map_sim_pars['combined_directory']): os.mkdir(meta.map_sim_pars['combined_directory'])

    print('looking for simulations on disk and organizing them (e.g. combining CMB+foregrounds)')
    list_of_sky_sim_folders = glob.glob(os.path.join(meta.map_sim_pars['external_sky_sims'], '*'))
    if not list_of_sky_sim_folders:
        print('the sky sim folder you provided looks empty!')
        sys.exit()

    list_of_noise_sim_folders = glob.glob(os.path.join(meta.map_sim_pars['external_noise_sims'], '*'))
    if not list_of_noise_sim_folders and not meta.general_pars['Nico_noise_combination']:
        print('the noise sim folder you provided looks empty!')
        sys.exit()
    if meta.general_pars['Nico_noise_combination']:
        print('we will combine white and one over f noise in map_simulator')
        list_of_noise_sim_folders = ['']*args.Nsims

    list_of_combined_directories = [] 
    for i_sim in range(args.Nsims): #TODO: do only one sim at a time? 
        print('creating following repo: ', os.path.join(meta.map_sim_pars['combined_directory'], str(i_sim).zfill(4)))
        list_of_combined_directories.append(os.path.join(meta.map_sim_pars['combined_directory'], str(i_sim).zfill(4)))
        for f in meta.general_pars['frequencies']:

            if not os.path.isfile(os.path.join(meta.map_sim_pars['combined_directory'], str(i_sim).zfill(4)+'/SO_SAT_'+str(f)+'_comb_'+str(i_sim).zfill(4)+'.fits')):

                if os.path.isfile(os.path.join(meta.map_sim_pars['combined_directory'], str(i_sim).zfill(4)+'/SO_SAT_'+str(f)+'_comb_'+str(i_sim).zfill(4)+'.fits')): continue

                # Getting dust:
                if meta.map_sim_pars['dust_model'] == 'Gaussian':
                    dust = hp.read_map( glob.glob(os.path.join(meta.map_sim_pars['external_sky_sims'], 'FG_20201207/gaussian/foregrounds/dust/'+str(i_sim).zfill(4)+'/SO_SAT_'+str(f)+'_dust_'+str(i_sim).zfill(4)+'*.fits'))[0], field=None)
                elif meta.map_sim_pars['dust_model'] == 'd0':
                    dust = hp.read_map( glob.glob(os.path.join(meta.map_sim_pars['external_sky_sims'], 'FG_20201207/realistic/d0s0/foregrounds/dust/SO_SAT_'+str(f)+'_dust_d0s0*.fits'))[0], field=None)
                elif meta.map_sim_pars['dust_model'] == 'd1':
                    dust = hp.read_map( glob.glob(os.path.join(meta.map_sim_pars['external_sky_sims'], 'FG_20201207/realistic/d1s1/foregrounds/dust/SO_SAT_'+str(f)+'_dust_d1s1*.fits'))[0], field=None)                    
                elif meta.map_sim_pars['dust_model'] == 'dm':
                    dust = hp.read_map( glob.glob(os.path.join(meta.map_sim_pars['external_sky_sims'], 'FG_20201207/realistic/dmsm/foregrounds/dust/SO_SAT_'+str(f)+'_dust_dmsm*.fits'))[0], field=None)                    
                elif meta.map_sim_pars['dust_model'] == 'd9':
                    dust = hp.read_map( glob.glob(os.path.join(meta.map_sim_pars['external_sky_sims'], 'FG_20220516/d9/foregrounds/dust/SO_SAT_'+str(f)+'_dust_d9*.fits'))[0], field=None)                    
                elif meta.map_sim_pars['dust_model'] == 'dh':
                    dust = hp.read_map( glob.glob(os.path.join(meta.map_sim_pars['external_sky_sims'], 'FG_20220516/dh/foregrounds/dust/SO_SAT_'+str(f)+'_dust_dh*.fits'))[0], field=None)                    
                elif meta.map_sim_pars['dust_model'] == 'd10':
                    dust = hp.read_map( glob.glob(os.path.join(meta.map_sim_pars['external_sky_sims'], 'FG_20220516/d10s5/foregrounds/dust/SO_SAT_'+str(f)+'_dust_d10s5*.fits'))[0], field=None)                    
                else:
                    print('WARNING: Dust model ', meta.map_sim_pars['dust_model'],' not recognized')

                # Getting synch
                if meta.map_sim_pars['sync_model'] == 'Gaussian':
                    synch = hp.read_map( glob.glob(os.path.join(meta.map_sim_pars['external_sky_sims'], 'FG_20201207/gaussian/foregrounds/synch/'+str(i_sim).zfill(4)+'/SO_SAT_'+str(f)+'_synch_'+str(i_sim).zfill(4)+'*.fits'))[0], field=None)
                elif meta.map_sim_pars['sync_model'] == 's0':
                    synch = hp.read_map( glob.glob(os.path.join(meta.map_sim_pars['external_sky_sims'], 'FG_20201207/realistic/d0s0/foregrounds/synch/SO_SAT_'+str(f)+'_synch_d0s0*.fits'))[0], field=None)
                elif meta.map_sim_pars['sync_model'] == 's1':
                    synch = hp.read_map( glob.glob(os.path.join(meta.map_sim_pars['external_sky_sims'], 'FG_20201207/realistic/d1s1/foregrounds/synch/SO_SAT_'+str(f)+'_synch_d1s1*.fits'))[0], field=None)
                elif meta.map_sim_pars['sync_model'] == 'sm':
                    synch = hp.read_map( glob.glob(os.path.join(meta.map_sim_pars['external_sky_sims'], 'FG_20201207/realistic/dmsm/foregrounds/synch/SO_SAT_'+str(f)+'_synch_dmsm*.fits'))[0], field=None)
                elif meta.map_sim_pars['sync_model'] == 's5':
                    synch = hp.read_map( glob.glob(os.path.join(meta.map_sim_pars['external_sky_sims'],'FG_20220516/d10s5/foregrounds/synch/SO_SAT_'+str(f)+'_synch_d10s5*.fits'))[0], field=None)
                elif meta.map_sim_pars['sync_model'] == 's7':
                    synch = hp.read_map( glob.glob(os.path.join(meta.map_sim_pars['external_sky_sims'], 'FG_20220516/s7/foregrounds/synch/SO_SAT_'+str(f)+'_synch_s7*.fits'))[0], field=None)                    
                else:
                    print('WARNING: Synchrotron model ', meta.map_sim_pars['sync_model'],' not recognized')

                if args.r_input == 0.0:
                    if args.AL_input == 1.0:
                        cmb = hp.read_map( glob.glob(os.path.join(meta.map_sim_pars['external_sky_sims'], 'CMB_r0_20201207/cmb/'+str(i_sim).zfill(4)+'/SO_SAT_'+str(f)+'_cmb_'+str(i_sim).zfill(4)+'*.fits'))[0], field=None) 
                    elif args.AL_input == 0.5:
                        cmb = hp.read_map( glob.glob(os.path.join(meta.map_sim_pars['external_sky_sims'], 'CMB_r_Alens_20211108/r0_Alens05/cmb/'+str(i_sim).zfill(4)+'/SO_SAT_'+str(f)+'_cmb_'+str(i_sim).zfill(4)+'*.fits'))[0], field=None) 
                elif args.r_input == 0.01:
                    if args.AL_input == 1.0:
                        cmb = hp.read_map( glob.glob(os.path.join(meta.map_sim_pars['external_sky_sims'], 'CMB_r_Alens_20211108/r001_Alens1/cmb/'+str(i_sim).zfill(4)+'/SO_SAT_'+str(f)+'_cmb_'+str(i_sim).zfill(4)+'*.fits'))[0], field=None) 
                    elif args.AL_input == 0.5:
                        cmb = hp.read_map( glob.glob(os.path.join(meta.map_sim_pars['external_sky_sims'], 'CMB_r_Alens_20211108/r001_Alens05/cmb/'+str(i_sim).zfill(4)+'/SO_SAT_'+str(f)+'_cmb_'+str(i_sim).zfill(4)+'*.fits'))[0], field=None) 
                else:
                    print('I do not know where to look for these input cosmo parameters ... ')
                    sys.exit()

                comb = dust + synch + cmb
                if not os.path.exists(os.path.join(meta.map_sim_pars['combined_directory'], str(i_sim).zfill(4))): os.mkdir(os.path.join(meta.map_sim_pars['combined_directory'], str(i_sim).zfill(4)))
                hp.write_map(os.path.join(meta.map_sim_pars['combined_directory'], str(i_sim).zfill(4)+'/SO_SAT_'+str(f)+'_comb_'+str(i_sim).zfill(4)+'.fits'), comb)


def get_sims(args):
    meta = BBmeta(args.globals)

    if meta.map_sim_pars['external_sky_sims']!='':
    
        if meta.map_sim_pars['combined_directory']=='': #TODO: this is not the right check, must see if the needed file is there
            loc_freq_map= get_combined_map_sims(args)
        else:
            freq_maps = np.zeros(3*len(meta.general_pars['frequencies']),hp.nside2npix(meta.general_pars['nside']))
            print('LOADING EXTERNAL SKY-ONLY MAPS')
            for f in range(len(instrument.frequency)):
                print('loading combined foregrounds map for frequency ', str(int(instrument.frequency[f])))
                loc_freq_map = hp.read_map(glob.glob(os.path.join(meta.map_sim_pars['combined_directory'],
                                                                  'SO_SAT_'+str(int(instrument.frequency[f]))+'_comb_*.fits'))[0],
                                                                   field=None)
                NSIDE_INPUT_MAP = hp.npix2nside(len(loc_freq_map[0]))
                alms = hp.map2alm(loc_freq_map, lmax=3*meta.general_pars['nside'])
                Bl_gauss_pix = hp.gauss_beam( hp.nside2resol(meta.general_pars['nside']), lmax=2*meta.general_pars['nside'])        
                for alm_ in alms: hp.almxfl(alm_, Bl_gauss_pix, inplace=True)             
                freq_maps[3*f:3*(f+1),:] = hp.alm2map(alms, meta.general_pars['nside'])
                del loc_freq_map

            print('f=', f, ' freq_maps = ', freq_maps[3*f:3*(f+1),:])
    else:
        d_config = meta.map_sim_pars['dust_model']
        s_config = meta.map_sim_pars['sync_model']
        c_config = meta.map_sim_pars['cmb_model']

        # performing the CMB simulation with synfast
        if meta.map_sim_pars['cmb_sim_no_pysm']:
            Cl_BB_prim = meta.map_sim_pars['r_input']*hp.read_cl(self.get_input('Cl_BB_prim_r1'))[2]
            Cl_lens = hp.read_cl(self.get_input('Cl_BB_lens'))
            l_max_lens = len(Cl_lens[0])
            Cl_BB_lens = meta.map_sim_pars['A_lens']*Cl_lens[2]
            Cl_TT = Cl_lens[0]
            Cl_EE = Cl_lens[1]
            Cl_TE = Cl_lens[3]

            sky = get_sky(meta.general_pars['nside'], d_config+s_config)
            Cl_BB = Cl_BB_prim[:l_max_lens] + Cl_BB_lens
            cmb_sky = hp.synfast([Cl_TT, Cl_EE, Cl_BB, Cl_TE, Cl_EE*0.0, Cl_EE*0.0], nside=meta.general_pars['nside'], new=True)
        else:
            sky = get_sky(meta.general_pars['nside'], c_config+d_config+s_config)   

        sky_CMB = get_sky(meta.general_pars['nside'], c_config)   
        sky_dust = get_sky(meta.general_pars['nside'], d_config)   
        sky_sync = get_sky(meta.general_pars['nside'], s_config)   

        channels = []
        for f_ in freqs:
            channels.append((np.array([f_-1,f_,f_+1]),np.array([0.0, 1.0, 0.0])))
        print('channels = ', channels)
        ###################################
        instrument = standardize_instrument(instrument_config)
        freq_maps = get_observation(instrument, sky) 

        if self.config['cmb_sim_no_pysm']:
            # adding CMB in this case
            for i in range(freq_maps.shape[0]):
                freq_maps[i,:,:] += cmb_sky[:,:]
            CMB_template_150GHz = cmb_sky
        else:
            CMB_template_150GHz = get_observation(instrument_150GHz, sky_CMB).reshape((3,noise_maps.shape[1]))
        dust_template_150GHz = get_observation(instrument_150GHz, sky_dust).reshape((3,noise_maps.shape[1]))
        sync_template_150GHz = get_observation(instrument_150GHz, sky_sync).reshape((3,noise_maps.shape[1]))

        # restructuration of the freq maps, of size {n_stokes x n_freqs, n_pix}
        freq_maps = freq_maps.reshape((3*len(meta.general_pars['frequencies']),hp.nside2npix(meta.general_pars['nside'])))
        NSIDE_INPUT_MAP = hp.npix2nside(len(freq_maps[0]))

    # adding noise
    if meta.general_pars['noise_option']=='no_noise': 
        pass
    elif meta.general_pars['external_noise_sims']!='' or meta.general_pars['Nico_noise_combination']:
        noise_maps = freq_maps*0.0
        print('LOADING EXTERNAL NOISE-ONLY MAPS')

        if meta.general_pars['Nico_noise_combination']:
            if meta.general_pars['knee_mode'] == 2 : knee_mode_loc = None
            else: knee_mode_loc = meta.general_pars['knee_mode']
            factors = compute_noise_factors(meta.general_pars['sensitivity_mode'], knee_mode_loc)

        for f in range(len(instrument.frequency)):
            print('loading noise map for frequency ', str(int(instrument.frequency[f])))

            if meta.general_pars['Nico_noise_combination']:
                noise_loc = combine_noise_maps(meta.map_sim_pars['isim'], instrument.frequency[f], factors)
            else:
                noise_loc = hp.read_map(glob.glob(os.path.join(meta.general_pars['external_noise_sims'],'SO_SAT_'+str(int(instrument.frequency[f]))+'_noise_FULL_*_white_20201207.fits'))[0], field=None)
            alms = hp.map2alm(noise_loc, lmax=3*meta.general_pars['nside'])
            Bl_gauss_pix = hp.gauss_beam( hp.nside2resol(meta.general_pars['nside']), lmax=2*meta.general_pars['nside'])        
            for alm_ in alms: hp.almxfl(alm_, Bl_gauss_pix, inplace=True)             
            noise_maps[3*f:3*(f+1),:] = hp.alm2map(alms, meta.general_pars['nside'])  

            if ((not meta.general_pars['no_inh']) and (meta.general_pars['Nico_noise_combination'])):
                # renormalize the noise map to take into account the effect of inhomogeneous noise
                print('rescaling the noise maps with hits map')
                nhits_nz = np.where(nhits!=0)[0]
                noise_maps[3*f:3*(f+1),nhits_nz] /= np.sqrt(nhits[nhits_nz]/np.max(nhits[nhits_nz]))


        freq_maps += noise_maps*binary_mask
    elif meta.general_pars['noise_option']=='white_noise':
        nlev_map = freq_maps*0.0
        for i in range(len(instrument.frequency)):
            nlev_map[3*i:3*i+3,:] = np.array([instrument.depth_i[i], instrument.depth_p[i], instrument.depth_p[i]])[:,np.newaxis]*np.ones((3,freq_maps.shape[-1]))
        nlev_map /= hp.nside2resol(meta.general_pars['nside'], arcmin=True)
        noise_maps = np.random.normal(freq_maps*0.0, nlev_map, freq_maps.shape)
        freq_maps += noise_maps*binary_mask
    else: 
        freq_maps += noise_maps*binary_mask

    freq_maps_unbeamed = freq_maps*1.0
    noise_maps_beamed = noise_maps*1.0


def get_maps(args):
    meta = BBmeta(args.globals)
    # get path from maps and import them

def pre_processing(args):
    meta = BBmeta(args.globals)
    IPython.embed()

    if meta.pre_proc_pars['common_beam_correction']!=0.0:
        print('  -> common beam correction: correcting for frequency-dependent beams and convolving with a common beam')
        Bl_gauss_common = hp.gauss_beam( np.radians(meta.pre_proc_pars['common_beam_correction']/60), lmax=2*meta.general_pars['nside'])        
        for f in range(len(meta.general_pars['frequencies'])):
            Bl_gauss_fwhm = hp.gauss_beam( np.radians(meta.pre_proc_pars['fwhm'][f]/60), lmax=2*meta.general_pars['nside'])
            alms = hp.map2alm(freq_maps[3*f:3*(f+1),:], lmax=3*meta.general_pars['nside'])
            for alm_ in alms:
                hp.almxfl(alm_, Bl_gauss_common/Bl_gauss_fwhm, inplace=True)             
            freq_maps[3*f:3*(f+1),:] = hp.alm2map(alms, meta.general_pars['nside'])   

            print('f=', f, ' freq_maps = ', freq_maps[3*f:3*(f+1),:])


    freq_maps[:,np.where(binary_mask==0)[0]] = hp.UNSEEN
    freq_maps_unbeamed[:,np.where(binary_mask==0)[0]] = hp.UNSEEN
    noise_maps[:,np.where(binary_mask==0)[0]] = hp.UNSEEN
    CMB_template_150GHz[:,np.where(binary_mask==0)[0]] = hp.UNSEEN



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

    if args.sims:
        freq_maps = get_sims(args)
    else:
        freq_maps = get_maps(args)

    pre_processing(args, freq_maps)