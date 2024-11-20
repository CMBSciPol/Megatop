# coding: utf-8
import healpy as hp
import scipy.sparse as ss
import numpy as np
import matplotlib.pyplot as plt
import os

##The CMBl are the validation with a CMB with only lensing BB. 
##The maps were produced at nside 512 with the pixel window function {pwf} and a smoothing of 30 arcmin.
##The TF sims the same, but I think you won’t need those probably. 
##Everything is in K.

glob_path = '/global/cfs/projectdirs/sobs/awg_bb/bbmaster_paper/'

paths = {
    'f': 'Foregrounds/',
    'CMBl': 'Validation_for_paper/CMBl_pwf_beam',
    'noise': 'Noise_forpaper/Atm_10m-reso/',   
}

SimNum = 10
filenames = {
    'f': 'd10s5_f090/filterbin_coadd-full_map.fits',
    'CMBl': '00{}/filterbin_coadd-full_map.fits'.format(SimNum),
    'noise': '00{}/filterbin_coadd-full_map.fits'.format(SimNum),   
}

#load observation matrix in TT only
#obsmat_path = os.path.join(glob_path,'obs_mat_nside128_fpthin8/obsmat_coadd-full.npz')
obsmat_TT = ss.load_npz('obs_TT.npz')

foreground = os.path.join(glob_path, paths['f'],filenames['f']) # Idk why it has to be like this format but don't change it :(
cmb = os.path.join(glob_path, paths['CMBl'],filenames['CMBl'])
noise = os.path.join(glob_path, paths['noise'],filenames['noise'])


map_f = hp.read_map(foreground, field = None) #These are E-only maps
map_cmb = hp.read_map(cmb, field = None)
map_n = hp.read_map(noise, field = None)

keys = ['CMB','noise','foreground']

nside = 128
npix = hp.nside2npix(nside)
total_map_T = np.zeros(npix, dtype=np.float64)

#Downgrade the maps:
for i,m in enumerate([map_cmb[0], map_n[0], map_f[0]]):
    m_nside128 = hp.ud_grade(m, 128)
    total_map_T += m_nside128

#Plot total map
if 1:
    plt.figure()
    mask = np.zeros(hp.nside2npix(128)) #Create a mask where 1 means masked.
    mask[np.where(total_map_T==0)]=1
    masked_m = hp.ma(total_map_T)
    masked_m.mask = mask
    #Plot the nside=128 total map
    hp.mollview(masked_m) 
    hp.graticule()
    plt.title ('Total map')
    plt.savefig('Total map pre obs.png')
    plt.close()

import time
start_time = time.time()
print(obsmat_TT.dtype)
map_out = obsmat_TT @ total_map_T
end_time = time.time()
print('Multiplication takes', end_time-start_time)