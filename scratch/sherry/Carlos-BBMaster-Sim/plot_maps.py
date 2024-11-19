# coding: utf-8
import healpy as hp
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

#load observation matrix 
obsmat_path = os.path.join(glob_path,'obs_mat_nside128_fpthin8/obsmat_coadd-full.npz')
obsmat = np.load(obsmat_path)
print(np.shape(obsmat))

foreground = os.path.join(glob_path, paths['f'],filenames['f']) # Idk why it has to be like this format but don't change it :(
cmb = os.path.join(glob_path, paths['CMBl'],filenames['CMBl'])
noise = os.path.join(glob_path, paths['noise'],filenames['noise'])


map_f = hp.fitsfunc.read_map(foreground) #These are temperature-only maps.
map_cmb = hp.fitsfunc.read_map(cmb)
map_n = hp.fitsfunc.read_map(noise)

keys = ['CMB','noise','foreground']
print(len(map_f))
for i,m in enumerate([map_cmb, map_n, map_f]):
    plt.figure()
    mask = np.zeros(hp.nside2npix(512)) #Create a mask where 1 means masked.
    mask[np.where(m==0)]=1
    masked_m = hp.ma(m)
    masked_m.mask = mask
    #Plot the nside=512 maps
    hp.mollview(masked_m) 
    hp.graticule()
    if i == 2:
        plt.title ('Foreground d10s5_f090 filterbin_coadd-full_map')
    else:
        plt.title ('the {}th Simulation for {}'.format(SimNum, keys[i]))
    plt.savefig('{}.png'.format(keys[i]))
    plt.close()


