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
import matplotlib.pyplot as plt

sys.path.append('/global/cfs/cdirs/sobs/users/krach/BBSims/NOISE_20201207/')
# from combine_noise import *

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

def MakeSims(args):
    meta = BBmeta(args.globals)
    d_config = meta.map_sim_pars['dust_model']
    s_config = meta.map_sim_pars['sync_model']
    # c_config = meta.map_sim_pars['cmb_model']

    # performing the CMB simulation with synfast
    if meta.map_sim_pars['cmb_sim_no_pysm']:
        print('Creating CMB map...')

        path_Cl_BB_lens = meta.get_fname_cls_fiducial_cmb('lensed')
        path_Cl_BB_prim_r1 = meta.get_fname_cls_fiducial_cmb('unlensed_scalar_tensor_r1')

        Cl_BB_prim = meta.map_sim_pars['r_input']*hp.read_cl(path_Cl_BB_prim_r1)[2]
        Cl_lens = hp.read_cl(path_Cl_BB_lens)

        if args.plots:
            print('Plotting Fiducial CMB spectra...')
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


        l_max_lens = len(Cl_lens[0])
        Cl_BB_lens = meta.map_sim_pars['A_lens']*Cl_lens[2]
        Cl_TT = Cl_lens[0]
        Cl_EE = Cl_lens[1]
        Cl_TE = Cl_lens[3]

        Cl_BB = Cl_BB_prim[:l_max_lens] + Cl_BB_lens
        cmb_sky = hp.synfast([Cl_TT, Cl_EE, Cl_BB, Cl_TE, Cl_EE*0.0, Cl_EE*0.0], 
                             nside=meta.general_pars['nside'], new=True)
    else:
        print('ERROR: CMB sims only handled using synfast on fiducial Cls')
        return 

    print('Initializing Instrument ...')
    import V3calc as V3

    #TODO: Optimize! Importing binary mask to compute fsky is a bit overkill... 
    binary_mask_path = meta.get_fname_mask('binary')

    binary_mask = hp.read_map(
        binary_mask_path,
        dtype=float)
    
    if meta.general_pars['nside'] != hp.npix2nside(binary_mask.shape[-1]):
        print('downgrading binary mask from nisde = ', hp.npix2nside(binary_mask.shape[-1]), 
              ' to nside = ',meta.general_pars['nside'])
        binary_mask = hp.ud_grade(binary_mask, nside_out=meta.general_pars['nside'])
        binary_mask[(binary_mask != 0) * (binary_mask != 1)] = 0
    
    fsky_binary = sum(binary_mask) / len(binary_mask)

    ell, N_ell_P_SA, Map_white_noise_levels = V3.so_V3_SA_noise(
        sensitivity_mode = meta.general_pars['sensitivity_mode'],
        one_over_f_mode = 2, # fixed to None since we only use white noise here
        SAC_yrs_LF = meta.general_pars['SAC_yrs_LF'], f_sky = fsky_binary, 
        ell_max = meta.general_pars['lmax'], delta_ell=1,
        beam_corrected=False, remove_kluge=False, CMBS4=''
    )

    instrument_config = {
        'frequency' : meta.general_pars['frequencies'],
        'depth_i' : Map_white_noise_levels/np.sqrt(2),
        'depth_p' : Map_white_noise_levels
        }

    instrument = standardize_instrument(instrument_config)

    print('Creating Pysm Fg maps...')
    sky = get_sky(meta.general_pars['nside'], d_config+s_config)

    fg_freq_maps = get_observation(instrument, sky, noise=False) 
    CMB_fg_freq_maps = fg_freq_maps + cmb_sky

    print('Beaming sky maps...')
    CMB_fg_freq_maps_beamed = []
    for f in range(len(meta.general_pars['frequencies'])):
        print('Beaming frequency channel:', meta.general_pars['frequencies'][f])
        CMB_fg_alms = hp.map2alm(CMB_fg_freq_maps[f], lmax=3*meta.general_pars['nside'])

        Bl_gauss_fwhm = hp.gauss_beam( np.radians(meta.pre_proc_pars['fwhm'][f]/60), lmax=3*meta.general_pars['nside'])
                
        for alm_ in CMB_fg_alms:
            hp.almxfl(alm_, Bl_gauss_fwhm, inplace=True)             

        CMB_fg_freq_maps_beamed.append(hp.alm2map(CMB_fg_alms, meta.general_pars['nside']) )        
    CMB_fg_freq_maps_beamed = np.array(CMB_fg_freq_maps_beamed)

    print('Creating noise maps...')
    if meta.general_pars['noise_option']=='white_noise':
        nlev_map = fg_freq_maps*0.0
        for i in range(len(instrument.frequency)):
            nlev_map[3*i:3*i+3,:] = np.array([instrument.depth_i[i], instrument.depth_p[i], instrument.depth_p[i]])[:,np.newaxis]*np.ones((3,fg_freq_maps.shape[-1]))
        nlev_map /= hp.nside2resol(meta.general_pars['nside'], arcmin=True)
        noise_maps = np.random.normal(fg_freq_maps*0.0, nlev_map, fg_freq_maps.shape)
    else:
        print('ERROR: Other noise cases not handled yet...')
        return
    
    CMB_fg_noise_freq_maps = CMB_fg_freq_maps_beamed + noise_maps
    return CMB_fg_noise_freq_maps
        

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
    print('DUMMY FUNCTION RETURN MAPS FULL OF ONES FOR TESTING NSIDE_INPUT = 512')
    print('SHAPE = (NFREQ, NSTOKES, NPIX)')
    freq_maps = np.ones((6,3,hp.nside2npix(512)))
    # freq_maps = np.ones((6,3,hp.nside2npix(meta.general_pars['nside'])))
    return freq_maps

def CommonBeamConvAndNsideModification(args, freq_maps, old_code = False):
    meta = BBmeta(args.globals)
    map_dimensions = len(freq_maps.shape)

    import time

    freq_maps_out = []

    # if meta.pre_proc_pars['common_beam_correction']!=0.0 and meta.general_pars['nside'] == hp.npix2nside(freq_maps.shape[-1]):
    if old_code:
        start1 = time.time()
        print('  -> common beam correction: correcting for frequency-dependent beams and convolving with a common beam')
        Bl_gauss_common = hp.gauss_beam( np.radians(meta.pre_proc_pars['common_beam_correction']/60), lmax=2*meta.general_pars['nside'])        
        for f in range(len(meta.general_pars['frequencies'])):
            Bl_gauss_fwhm = hp.gauss_beam( np.radians(meta.pre_proc_pars['fwhm'][f]/60), lmax=2*meta.general_pars['nside'])
            
            if map_dimensions == 2: # if maps are stored in (nstokes*nfreq, npix)
                alms = hp.map2alm(freq_maps[3*f:3*(f+1),:], lmax=3*meta.general_pars['nside']) 
            elif map_dimensions == 3: # if maps are stored in (nfreq, nstokes, npix)
                alms = hp.map2alm(freq_maps[f], lmax=3*meta.general_pars['nside'])
            else:
                print('freq_maps doesn\'t have the right number of dimensions, either 2 (nstokes*nfreq, npix), or 3 (nfreq, nstokes, npix)') 
                print('returning original freq_maps ...')
                return freq_maps
            
            for alm_ in alms:
                hp.almxfl(alm_, Bl_gauss_common/Bl_gauss_fwhm, inplace=True)             
            # freq_maps[3*f:3*(f+1),:] = hp.alm2map(alms, meta.general_pars['nside'])   
            freq_maps_out.append(hp.alm2map(alms, meta.general_pars['nside']) )
        print('time = ', time.time() - start1)
        
    # elif meta.pre_proc_pars['common_beam_correction']!=0.0 and meta.general_pars['nside'] != hp.npix2nside(freq_maps.shape[-1]):
    else:
    
        print('  -> common beam correction and NSIDE change: correcting for frequency-dependent beams, convolving with a common beam, modifying NSIDE and include effect of pixel window function')
        start2 = time.time()

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
            
            alm_in_T,alm_in_E,alm_in_B = hp.map2alm([cmb_in_T,cmb_in_Q,cmb_in_U],lmax=lmax_convolution,pol=True)
            # here lmax seems to play an important role            
            

            #change beam and wpix
            alm_out_T = hp.almxfl(alm_in_T,sm_corr_T)
            alm_out_E = hp.almxfl(alm_in_E,sm_corr_P)
            alm_out_B = hp.almxfl(alm_in_B,sm_corr_P)

            #alm-->map
            cmb_out_T,cmb_out_Q,cmb_out_U = hp.alm2map([alm_out_T,alm_out_E,alm_out_B],meta.general_pars['nside'],
                                                       lmax=lmax_convolution,pixwin=False,fwhm=0.0,pol=True,verbose=False) 
            # a priori all the options are set to there default, even lmax which is computed wrt input alms
            marco_out_map = np.array([cmb_out_T,cmb_out_Q,cmb_out_U])
            freq_maps_out.append(marco_out_map)
        print('time = ', time.time() - start2)
        
            
    # else:
    #     print('case not handled yet, if you want to change nside only without common beam or something else it is not yet implemented')
    #     freq_maps_out = freq_maps
    #   TODO: 2 dim
    return np.array(freq_maps_out)
    # freq_maps[:,np.where(binary_mask==0)[0]] = hp.UNSEEN
    # freq_maps_unbeamed[:,np.where(binary_mask==0)[0]] = hp.UNSEEN
    # noise_maps[:,np.where(binary_mask==0)[0]] = hp.UNSEEN
    # CMB_template_150GHz[:,np.where(binary_mask==0)[0]] = hp.UNSEEN

def ApplyBinaryMask(args, freq_maps, use_UNSEEN = False):
    meta = BBmeta(args.globals)
    binary_mask_path = meta.get_fname_mask('binary')

    binary_mask = hp.read_map(
        binary_mask_path,
        dtype=float)
    
    if meta.general_pars['nside'] != hp.npix2nside(binary_mask.shape[-1]):
        print('downgrading binary mask from nisde = ', hp.npix2nside(binary_mask.shape[-1]), 
              ' to nside = ',meta.general_pars['nside'])
        binary_mask = hp.ud_grade(binary_mask, nside_out=meta.general_pars['nside'])
        binary_mask[(binary_mask != 0) * (binary_mask != 1)] = 0

    # binary_mask[np.where(nhits<1e-6)[0]] = 0.0


    if use_UNSEEN:
        freq_maps[:,np.where(binary_mask==0)[0]] = hp.UNSEEN

    else:
        freq_maps *= binary_mask

    return freq_maps



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

    # if args.sims and args.plots:
    #     warnings.warn("Both --sims and --plot are set to True. "
    #                   "Too many plots will be generated. "
    #                   "Set --plot to False")
    #     args.plots = False

    if args.sims:
        print('Simulating maps ...')
        freq_maps = MakeSims(args)
    else:
        print('Importing maps ...')
        freq_maps = get_maps(args)
    

    freq_maps_common_beamed = CommonBeamConvAndNsideModification(args, freq_maps)

    freq_maps_common_beamed_masked = ApplyBinaryMask(args, freq_maps_common_beamed)
    # freq_maps_unbeamed_masked = ApplyBinaryMask(args, freq_maps)

    meta = BBmeta(args.globals)

    print('saving maps ...')
    #  saving the sims 
    if args.sims:
        np.save(os.path.join(meta.output_dirs['root'], meta.output_dirs['sims_directory'], 'freq_maps.npy' ),
                freq_maps )

    np.save(os.path.join(meta.output_dirs['root'], meta.output_dirs['pre_process_directory'], 'freq_maps_common_beamed_masked.npy' ),
            freq_maps_common_beamed_masked )
    # np.save(os.path.join(meta.output_dirs['root'], meta.output_dirs['pre_process_directory'], 'freq_maps_unbeamed_masked.npy' ),
    #         freq_maps_unbeamed_masked )    

    IPython.embed()