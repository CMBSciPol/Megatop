import numpy as np
import healpy as hp
import IPython
import matplotlib.pyplot as plt
import os

H_PLANCK = 6.6260755e-34
K_BOLTZ = 1.380658e-23
T_CMB = 2.72548
def thermo2rj(nu):
    x = H_PLANCK*nu*1.e9/(K_BOLTZ*T_CMB)
    return x**2 * np.exp(x) / (np.exp(x) - 1.0)**2
# since I did the dust at f090, that is the reference
def sed_dust(nu, beta, Tdust):
    sed_fact_353 = (93e9)**(beta+1) / (np.exp(H_PLANCK*93e9/(K_BOLTZ*Tdust))-1) / thermo2rj(93.0) ; sed_fact_nu  = (nu * 1e9)**(beta+1) / (np.exp(H_PLANCK*nu*1e9/(K_BOLTZ*Tdust))-1) / thermo2rj(nu)
    return sed_fact_nu / sed_fact_353



def apply_beam_correction(maps, beam_out, lmax):
    almT, almE, almB = hp.map2alm(maps, lmax=lmax, pol=True, iter=10)

    alm_out_T = hp.almxfl(almT,beam_out)
    alm_out_E = hp.almxfl(almE,beam_out)
    alm_out_B = hp.almxfl(almB,beam_out)

    outmap_T, outmap_Q, outmap_U = hp.alm2map([alm_out_T,alm_out_E,alm_out_B], hp.npix2nside(maps.shape[-1]),
                                                lmax=lmax ,pixwin=False,fwhm=0.0,pol=True) 

    return np.array([outmap_T, outmap_Q, outmap_U])


def plot_all_maps(map_dict, output, file_name):
    '''
    Plot all maps in dictionary

    Parameters:
    -----------
    map_dict : dict
        Dictionary containing T,Q,U component maps to be plotted for a given frequency
    output : str
        Path to save the plots

    Returns:
    --------
    None
    '''
    # ploting Q and U maps for all entries in dictionary
    line_number = len(map_dict.keys())
    stokes = ['T', 'Q', 'U']
    j = 1  
    plt.figure(figsize=(6,12))
    for key in map_dict.keys():
        for i in range(1,3):
            map_plot =  map_dict[key][i]
            map_plot[map_plot==0] = hp.UNSEEN  
            hp.mollview(map_plot, title=f'{key} {stokes[i]}', sub=(line_number,2,j)) #, norm='hist')
            j += 1
    os.makedirs(os.path.join(output, 'plots'), exist_ok=True)
    plt.savefig(os.path.join(output, 'plots',f'{file_name}.png'), dpi=400, bbox_inches='tight')
    plt.close()

def plot_all_spectra(spectra_dict, output, file_name):
    '''
    Plot all spectra in dictionary

    Parameters:
    -----------
    spectra_dict : dict
        Dictionary containing T,Q,U component spectra to be plotted for a given frequency
    output : str
        Path to save the plots

    Returns:
    --------
    None
    '''
    # ploting Q and U maps for all entries in dictionary
    line_number = len(spectra_dict.keys())
    stokes = ['TT', 'EE', 'BB', 'TE']
    j = 1  

    fig, ax = plt.subplots( line_number, figsize=(6,12))
    if line_number == 1:
        ax = [ax] # to avoid problems with indexing when only one key is present
    for i, key in enumerate(spectra_dict.keys()):
        for j in range(1,3):
            ax[i].plot(spectra_dict[key][j], label=stokes[j])
            ax[i].set_yscale('log')
            ax[i].set_xscale('log')
            ax[i].legend()
            ax[i].set_title(key)
    
    os.makedirs(os.path.join(output, 'plots'), exist_ok=True)
    plt.savefig(os.path.join(output, 'plots',f'{file_name}.png'), dpi=400, bbox_inches='tight')
    plt.close()

def main():
    plot_maps = True
    plot_spectra = True
    add_pixwin = False

    beta_dust = 1.54
    Tdust = 20.0

    nside = 512
    lmax = 3*nside - 1
    Nmc = 100

    # REMEMBER THE GAUSSIAN DUST MAPS ARE IN uK, SO WE NEED TO TRASNFORM TO K FIRST !!!

    beam_30 = hp.gauss_beam(np.radians(0.5), lmax=3*512-1, pol=True)
    beam_17 = hp.gauss_beam(np.radians(17.0/60.0), lmax=3*512-1, pol=True)
    beam_5 = hp.gauss_beam(np.radians(5.0/60.0), lmax=3*512-1, pol=True)
    beams_array = [beam_30,beam_17,beam_5,beam_5]
    # print('WARNING: Using the same beam for all frequencies')
    # beams_array = [beam_30,beam_30,beam_30,beam_30]
    # beams_array = [beam_30,beam_30,beam_30,beam_30,beam_17,beam_17,beam_17,beam_17,beam_5,beam_5]
    freq_array = ['f090_full','f150_full','f353_A_full','f353_B_full']
    # freq_array = ['f090_hm1','f090_hm2','f090_hm3','f090_hm4','f150_hm1','f150_hm2','f150_hm3','f150_hm4','353_A_full','353_B_full']
    freq_values = [93.,145.,353.,353.]

    # mask = hp.read_map('toast/data/mask_SAT_FirstDayEveryMonth_apo5.0_fpthin8_nside512.fits', dtype=np.double)
    print('')

    root = '/global/cfs/projectdirs/sobs/awg_bb/bbmaster_paper/'
    output_root = '/pscratch/sd/j/jost/SO_MEGATOP/CarlosInput/'
    rank = 0 # can be MPI'sed
    sim_id = rank

    dust_model = 'Gaussian_f090'

    output = output_root+f'{dust_model}_{sim_id:04}/'
    # output = output_root+f'{dust_model}_{sim_id:04}_samebeam30/'
    os.makedirs(output, exist_ok=True)
    os.makedirs(os.path.join(output, 'noise'), exist_ok=True)
    list_CMBplusNoise_maps_planck_splits = []
    for i, band in enumerate(freq_array):
        print('Band:', band)

        if '353' not in band:
            cmb_map_band = hp.read_map(root+f'Validation_for_paper/CMBl_pwf_beam/{sim_id:04}/filterbin_coadd-full_map.fits', field=(0,1,2), dtype=np.double)
            dust_map_band = hp.read_map(root+f'Foregrounds/{dust_model}/{sim_id:04}/filterbin_coadd-full_map.fits', field=(0,1,2), dtype=np.double)
            if band == 'f090_full':        
                noise_map_band = hp.read_map(root+f'Noise_forpaper/Atm_10m-reso/{sim_id:04}/filterbin_coadd-full_map.fits', field=(0,1,2), dtype=np.double)
                noise_map_band *= 1e6 # from K to uK

            elif band == 'f150_full':
                noise_map_band = hp.read_map(root+f'Noise_forpaper/Atm_10m-reso_f150/{sim_id:04}/filterbin_coadd-full_map.fits', field=(0,1,2), dtype=np.double)
                # noise_map_band_hm1 = hp.read_map(root+f'Noise_forpaper/Atm_10m-reso_f150/{sim_id:04}/filterbin_coadd-hm1_map.fits', field=(0,1,2), dtype=np.double)
                # noise_map_band_hm2 = hp.read_map(root+f'Noise_forpaper/Atm_10m-reso_f150/{sim_id:04}/filterbin_coadd-hm2_map.fits', field=(0,1,2), dtype=np.double)
                # noise_map_band_hm3 = hp.read_map(root+f'Noise_forpaper/Atm_10m-reso_f150/{sim_id:04}/filterbin_coadd-hm3_map.fits', field=(0,1,2), dtype=np.double)
                # noise_map_band_hm4 = hp.read_map(root+f'Noise_forpaper/Atm_10m-reso_f150/{sim_id:04}/filterbin_coadd-hm4_map.fits', field=(0,1,2), dtype=np.double)
                # noise_map_band = noise_map_band_hm1 + noise_map_band_hm2 + noise_map_band_hm3 + noise_map_band_hm4
                noise_map_band *= 1e6 # from K to uK

            else:
                print('ERROR: No noise maps for {} band'.format(band))
                exit()
            # this is a mask that I created on the fly, it is not necesseraly the same as the one used in the simulations
            # any pixels that is 0 in cmb, dust or noise is masked:
            mask_on_the_fly = np.ones(cmb_map_band.shape[-1])
            # We assume the mask is the same for T,Q,U
            mask_on_the_fly[cmb_map_band[0]==0] = 0
            mask_on_the_fly[dust_map_band[0]==0] = 0
            mask_on_the_fly[noise_map_band[0]==0] = 0
            
            if band != 'f090_full':
                dust_map_band = dust_map_band * sed_dust(freq_values[i], beta_dust, Tdust)            
                beam_correction = (beams_array[i][:,1]/beams_array[0][:,1])
                if add_pixwin:
                    print('WARNING: Applying pixel window function to the beam correction (not sure if it makes sense...)')
                    pixwin = hp.pixwin(nside)[:lmax+1]
                    beam_correction *= pixwin
                sky_map_masked = (cmb_map_band+dust_map_band)*mask_on_the_fly

                corrected_maps = apply_beam_correction(sky_map_masked, beam_correction, lmax)
                corrected_maps = corrected_maps * mask_on_the_fly
            else:
                corrected_maps = (cmb_map_band+dust_map_band)*mask_on_the_fly
            
            # From K to uK:
            corrected_maps *= 1e6
            
            total_maps = corrected_maps + noise_map_band*mask_on_the_fly

            if plot_maps or plot_spectra:
                map_dict = {'cmb':cmb_map_band, 'dust':dust_map_band, 'noise':noise_map_band, 'corrected':corrected_maps, 'total':total_maps}
        elif '353' in band:
            split = band[band.find('_')+1]
            cmb_map_band = hp.read_map(root+f'Validation_for_paper/Planck_CMB-noise_353{split}_pwf_beam/{sim_id:04}/filterbin_coadd-full_map.fits', field=(0,1,2), dtype=np.double)
            list_CMBplusNoise_maps_planck_splits.append(cmb_map_band)
            dust_map_band  = hp.read_map(root+f'Foregrounds/{dust_model}/{sim_id:04}/filterbin_coadd-full_map.fits',field=(0,1,2), dtype=np.double)
            dust_map_band = dust_map_band * sed_dust(freq_values[i], beta_dust, Tdust)
            
            mask_on_the_fly = np.ones(cmb_map_band.shape[-1])
            # We assume the mask is the same for T,Q,U
            mask_on_the_fly[cmb_map_band[0]==0] = 0
            mask_on_the_fly[dust_map_band[0]==0] = 0
            # mask_on_the_fly[noise_map_band[0]==0] = 0
            beam_correction = (beams_array[i][:,1]/beams_array[0][:,1])

            if add_pixwin:
                print('WARNING: Applying pixel window function to the beam correction (not sure if it makes sense...)')
                pixwin = hp.pixwin(nside)[:lmax+1]
                beam_correction *= pixwin
            
            corrected_maps = apply_beam_correction(dust_map_band* mask_on_the_fly, beam_correction, lmax)
            corrected_maps = corrected_maps * mask_on_the_fly
            total_maps = corrected_maps + cmb_map_band * mask_on_the_fly
            # From K to uK:
            total_maps = total_maps * 1e6

            if plot_maps or plot_spectra:
                map_dict = {'cmb+noise(?)':cmb_map_band, 'dust':dust_map_band*mask_on_the_fly, 'corrected':corrected_maps, 'total':total_maps}
        else:
            print('ERROR: No maps for {} band'.format(band))
            exit()

        hp.write_map(output+f'{band}_map.fits', total_maps, dtype=np.double, overwrite=True)
        print('Map written to:', output+f'{band}_map.fits')
        
        if '353' not in band:
            hp.write_map(output+f'noise/{band}_noise_map.fits', noise_map_band, dtype=np.double, overwrite=True)
            print('Noise map written to:', output+f'noise/{band}_noise_map.fits')

        if plot_maps:
            plot_all_maps(map_dict, output, f'check_maps_{band}')
        
        if plot_spectra:
            spectra_dict = {}
            for key in map_dict.keys():
                spectra_dict[key] = hp.anafast(map_dict[key]*mask_on_the_fly, lmax=lmax)
            plot_all_spectra(spectra_dict, output, f'check_spectra_{band}')

    #  Computing noise map for Planck with the difference between the two splits
    noise_353 = list_CMBplusNoise_maps_planck_splits[1] - list_CMBplusNoise_maps_planck_splits[0]
    noise_353 *= 1e6  # from K to uK 
    noise_353 /= np.sqrt(2)  # and applying the sqrt(2) factor to get the noise of a single split

    hp.write_map(output+f'noise/f353_noise_map_diff_splits.fits', noise_353, dtype=np.double, overwrite=True)
    map_dict_noise_split = {'Noise diff split 353': noise_353}
    if plot_maps:
        plot_all_maps(map_dict_noise_split, output, 'check_maps_Noise_diff_split_353')
    
    if plot_spectra:
        spectra_dict = {}
        for key in map_dict_noise_split.keys():
            spectra_dict[key] = hp.anafast(map_dict_noise_split[key]*mask_on_the_fly, lmax=lmax)
        plot_all_spectra(spectra_dict, output, f'check_spectra_Noise_diff_split_353')    


if __name__ == '__main__':
    main()


