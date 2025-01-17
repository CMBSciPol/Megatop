import pymaster as nmt
#import matplotlib.pyplot as pl
import numpy as np
import healpy as hp
from mpi4py import MPI

comm = MPI.COMM_WORLD
size_mpi = comm.Get_size()
rank = comm.Get_rank()

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

beta_dust = 1.54
Tdust = 20.0

nside = 512
lmax = 3*nside - 1
Nmc = 100

# REMEMBER THE GAUSSIAN DUST MAPS ARE IN uK, SO WE NEED TO TRASNFORM TO K FIRST !!!

beam_30 = hp.gauss_beam(np.radians(0.5), lmax=3*512-1, pol=True)
beam_17 = hp.gauss_beam(np.radians(17.0/60.0), lmax=3*512-1, pol=True)
beam_5 = hp.gauss_beam(np.radians(5.0/60.0), lmax=3*512-1, pol=True)
beams_array = [beam_30,beam_30,beam_30,beam_30,beam_17,beam_17,beam_17,beam_17,beam_5,beam_5]
freq_array = ['f090_hm1','f090_hm2','f090_hm3','f090_hm4','f150_hm1','f150_hm2','f150_hm3','f150_hm4','353_A_full','353_B_full']
freq_values = [93.,93.,93.,93.,145.,145.,145.,145.,353.,353.]

mask = hp.read_map('toast/data/mask_SAT_FirstDayEveryMonth_apo5.0_fpthin8_nside512.fits', dtype=np.double)

# first I need the binning scheme and the mask
# this is mcmer.py
bins = nmt.NmtBin.from_nside_linear(nside, nlb=10, is_Dell=False)
ell_bins = bins.get_effective_ells()
ell_ = np.arange(lmax+1)
Nbins = bins.get_n_bands()
field_spin2 = nmt.NmtField(mask, None, spin=2, purify_b=False)
w = nmt.NmtWorkspace()
w.compute_coupling_matrix(field_spin2, field_spin2, bins,)
nspec = 4
mcm = np.transpose(w.get_coupling_matrix().reshape([lmax+1, nspec, lmax+1, nspec]), axes=[1, 0, 3, 2])
binner = np.array([bins.bin_cell(np.array([cl]))[0] for cl in np.eye(lmax+1)]).T
mcm_binned = np.einsum('ij,kjlm->kilm', binner, mcm)

array = np.load('toast/output/PCell_filt_FirstDayEveryMonth_nside512_fpthin8_pwf_beam_crossCL.npy')
cl0 = np.zeros(3*nside)
ell_arr = np.arange(lmax+1)
cl_ = 1/(ell_arr+10)**2
cls_PL = np.array([cl_,cl_,cl_,cl_])
size = 4


trans_ = np.zeros((len(freq_array),len(freq_array),4,4,Nbins))
inv_couplings = np.zeros((len(freq_array),len(freq_array),4*Nbins,4*Nbins))

# We do the TF once here
for i in range(len(freq_array)):
    for j in range(i, len(freq_array)):
        #if i==j: continue
        pcls_mat_filtered_mean = np.mean(array[i,j], axis=0)
        if False:
            pcls_mat_unfiltered_mean = np.mean(array2[i,j], axis=0)
            cct_inv = np.transpose(np.linalg.inv(np.transpose(np.einsum('jil,jkl->ikl',pcls_mat_unfiltered_mean,pcls_mat_unfiltered_mean),axes=[2, 0, 1])), axes=[1, 2, 0])
            trans_[i,j] = np.einsum('ijl,jkl->kil', cct_inv,np.einsum('jil,jkl->ikl',pcls_mat_unfiltered_mean,pcls_mat_filtered_mean))
        
        # THIS IS WITH THEORY
        pcls_mat_unfiltered_mean = np.array([bins.bin_cell(np.einsum('ijkl,kl', mcm, np.array([cls_PL[0], cl0, cl0, cl0]))),bins.bin_cell(np.einsum('ijkl,kl', mcm,np.array([cl0, cls_PL[1], cl0, cl0]))),bins.bin_cell(np.einsum('ijkl,kl', mcm,np.array([cl0, cl0, cls_PL[2], cl0]))),bins.bin_cell(np.einsum('ijkl,kl', mcm,np.array([cl0, cl0, cl0, cls_PL[3]])))])
        cct_invs = np.transpose( np.linalg.inv( np.transpose( np.einsum('jil,jkl->ikl', pcls_mat_unfiltered_mean, pcls_mat_unfiltered_mean), axes=[2, 0, 1])), axes=[1, 2, 0])
        trans_[i,j] = np.einsum('ijl,jkl->kil', cct_invs, np.einsum('jil,jkl->ikl', pcls_mat_unfiltered_mean, pcls_mat_filtered_mean))
        tbmcm = np.einsum('ijk,jklm->iklm', trans_[i,j], mcm_binned)
        # Fully binned coupling matrix (including filtering)
        btbmcm = np.transpose(np.array([np.sum(tbmcm[:, :, :, bins.get_ell_list(i)],axis=-1) for i in range(Nbins)]), axes=[1, 2, 3, 0])
        ibtbmcm = np.linalg.inv(btbmcm.reshape([size*Nbins, size*Nbins]))
        winflat = np.dot(ibtbmcm, tbmcm.reshape([size*Nbins, size*(lmax+1)]))
        wcal_inv = ibtbmcm.reshape([size, Nbins, size, Nbins])
        bpw_windows = winflat.reshape([size, Nbins, size, lmax+1])
        couplings_nobeam = {}
        couplings_nobeam[f"bp_win"] = bpw_windows
        couplings_nobeam[f"inv_coupling"] = wcal_inv
        inv_couplings[i,j] = couplings_nobeam[f"inv_coupling"].reshape([4*Nbins, 4*Nbins])
# We save the transfer function to file
if rank==0:
    np.savez('output/TF_FirstDayEveryMonth_Full_nside512_fpthin8_CrossCL_pwf_beam', tf=trans_, ells=ell_bins, inv_couplings=inv_couplings)

indices_split = np.array_split(range(Nmc), size_mpi)
PCell_obs = np.zeros((len(indices_split[rank]), len(freq_array), len(freq_array), 4, Nbins))

for id_sim_rel, id_sim in enumerate(indices_split[rank]):
    for i in range(len(freq_array)):
        band_i = freq_array[i]
        if band_i == 'f090_hm1':
            cmb_i = hp.read_map('toast/output/Validation_for_paper/CMBl_pwf_beam/%04i/filterbin_coadd-hm1_map.fits'%(id_sim), field=(1,2), dtype=np.double)
            noise_i = hp.read_map('toast/output/Noise_for_paper/Atm_10m-reso/%04i/filterbin_coadd-hm1_map.fits'%(id_sim), field=(1,2), dtype=np.double)
            dust_i = hp.read_map('toast/output/Foregrounds/Gaussian/%04i/filterbin_coadd-hm1_map.fits'%(id_sim), field=(1,2), dtype=np.double)
        elif band_i == 'f150_hm1':
            cmb_i = hp.read_map('toast/output/Validation_for_paper/CMBl_pwf_beam/%04i/filterbin_coadd-hm1_map.fits'%(id_sim), field=(1,2), dtype=np.double)
            noise_i = hp.read_map('toast/output/Noise_for_paper/Atm_10m-reso_f150/%04i/filterbin_coadd-hm1_map.fits'%(id_sim), field=(1,2), dtype=np.double)
            dust_i = hp.read_map('toast/output/Foregrounds/Gaussian/%04i/filterbin_coadd-hm1_map.fits'%(id_sim), field=(1,2), dtype=np.double)
        elif band_i == 'f090_hm2':
            cmb_i   = hp.read_map('toast/output/Validation_for_paper/CMBl_pwf_beam/%04i/filterbin_coadd-hm2_map.fits'%(id_sim), field=(1,2), dtype=np.double)
            noise_i = hp.read_map('toast/output/Noise_for_paper/Atm_10m-reso/%04i/filterbin_coadd-hm2_map.fits'%(id_sim), field=(1,2), dtype=np.double)
            dust_i  = hp.read_map('toast/output/Foregrounds/Gaussian/%04i/filterbin_coadd-hm2_map.fits'%(id_sim), field=(1,2), dtype=np.double)
        elif band_i == 'f150_hm2':
            cmb_i   = hp.read_map('toast/output/Validation_for_paper/CMBl_pwf_beam/%04i/filterbin_coadd-hm2_map.fits'%(id_sim), field=(1,2), dtype=np.double)
            noise_i = hp.read_map('toast/output/Noise_for_paper/Atm_10m-reso_f150/%04i/filterbin_coadd-hm2_map.fits'%(id_sim), field=(1,2), dtype=np.double)
            dust_i  = hp.read_map('toast/output/Foregrounds/Gaussian/%04i/filterbin_coadd-hm2_map.fits'%(id_sim), field=(1,2), dtype=np.double)
        elif band_i == 'f090_hm3':
            cmb_i   = hp.read_map('toast/output/Validation_for_paper/CMBl_pwf_beam/%04i/filterbin_coadd-hm3_map.fits'%(id_sim), field=(1,2), dtype=np.double)
            noise_i = hp.read_map('toast/output/Noise_for_paper/Atm_10m-reso/%04i/filterbin_coadd-hm3_map.fits'%(id_sim), field=(1,2), dtype=np.double)
            dust_i  = hp.read_map('toast/output/Foregrounds/Gaussian/%04i/filterbin_coadd-hm3_map.fits'%(id_sim), field=(1,2), dtype=np.double)
        elif band_i == 'f150_hm3':
            cmb_i   = hp.read_map('toast/output/Validation_for_paper/CMBl_pwf_beam/%04i/filterbin_coadd-hm3_map.fits'%(id_sim), field=(1,2), dtype=np.double)
            noise_i = hp.read_map('toast/output/Noise_for_paper/Atm_10m-reso_f150/%04i/filterbin_coadd-hm3_map.fits'%(id_sim), field=(1,2), dtype=np.double)
            dust_i  = hp.read_map('toast/output/Foregrounds/Gaussian/%04i/filterbin_coadd-hm3_map.fits'%(id_sim), field=(1,2), dtype=np.double)
        elif band_i == 'f090_hm4':
            cmb_i   = hp.read_map('toast/output/Validation_for_paper/CMBl_pwf_beam/%04i/filterbin_coadd-hm4_map.fits'%(id_sim), field=(1,2), dtype=np.double)
            noise_i = hp.read_map('toast/output/Noise_for_paper/Atm_10m-reso/%04i/filterbin_coadd-hm4_map.fits'%(id_sim), field=(1,2), dtype=np.double)
            dust_i  = hp.read_map('toast/output/Foregrounds/Gaussian/%04i/filterbin_coadd-hm4_map.fits'%(id_sim), field=(1,2), dtype=np.double)
        elif band_i == 'f150_hm4':
            cmb_i   = hp.read_map('toast/output/Validation_for_paper/CMBl_pwf_beam/%04i/filterbin_coadd-hm4_map.fits'%(id_sim), field=(1,2), dtype=np.double)
            noise_i = hp.read_map('toast/output/Noise_for_paper/Atm_10m-reso_f150/%04i/filterbin_coadd-hm4_map.fits'%(id_sim), field=(1,2), dtype=np.double)
            dust_i  = hp.read_map('toast/output/Foregrounds/Gaussian/%04i/filterbin_coadd-hm4_map.fits'%(id_sim), field=(1,2), dtype=np.double)
        elif band_i == '353_A_full':
            cmb_i = hp.read_map('toast/output/Validation_for_paper/Planck_CMB-noise_353A_pwf_beam/%04i/filterbin_coadd-full_map.fits'%(id_sim), field=(1,2), dtype=np.double)
            dust_i  = hp.read_map('toast/output/Foregrounds/Gaussian/%04i/filterbin_coadd-full_map.fits'%(id_sim),field=(1,2), dtype=np.double)
        elif band_i == '353_B_full':
            cmb_i = hp.read_map('toast/output/Validation_for_paper/Planck_CMB-noise_353B_pwf_beam/%04i/filterbin_coadd-full_map.fits'%(id_sim), field=(1,2), dtype=np.double)
            dust_i  = hp.read_map('toast/output/Foregrounds/Gaussian/%04i/filterbin_coadd-full_map.fits'%(id_sim),field=(1,2), dtype=np.double)
        
        sed_factor_i = sed_dust(freq_values[i], beta_dust, Tdust)
        if '353' in band_i:
            # The cmb already includes noise, I only need to correct the dust beam
            field_spin2_beam_i = nmt.NmtField(mask, [dust_i[0]*sed_factor_i, dust_i[1]*sed_factor_i])
            field_spin2_nobeam_i = nmt.NmtField(mask, [cmb_i[0], cmb_i[1]])
        else:
            # We are in f090 or f150, so cmb and dust have the same beam and I need to correct for it
            field_spin2_beam_i = nmt.NmtField(mask, [cmb_i[0] + dust_i[0]*sed_factor_i, cmb_i[1] + dust_i[1]*sed_factor_i])
            field_spin2_nobeam_i = nmt.NmtField(mask, [noise_i[0], noise_i[1]])
        beam_correction_i = (beams_array[i][:,1]/beams_array[0][:,1]) # this correction factor is needed for the cmb+dust
        # ---------------
        for j in range(i, len(freq_array)):
            if i==j: continue
            band_j = freq_array[j]
            if band_j == 'f090_hm1':
                cmb_j   = hp.read_map('toast/output/Validation_for_paper/CMBl_pwf_beam/%04i/filterbin_coadd-hm1_map.fits'%(id_sim), field=(1,2), dtype=np.double)
                noise_j = hp.read_map('toast/output/Noise_for_paper/Atm_10m-reso/%04i/filterbin_coadd-hm1_map.fits'%(id_sim), field=(1,2), dtype=np.double)
                dust_j  = hp.read_map('toast/output/Foregrounds/Gaussian/%04i/filterbin_coadd-hm1_map.fits'%(id_sim), field=(1,2), dtype=np.double)
            elif band_j == 'f150_hm1':
                cmb_j = hp.read_map('toast/output/Validation_for_paper/CMBl_pwf_beam/%04i/filterbin_coadd-hm1_map.fits'%(id_sim), field=(1,2), dtype=np.double)
                noise_j = hp.read_map('toast/output/Noise_for_paper/Atm_10m-reso_f150/%04i/filterbin_coadd-hm1_map.fits'%(id_sim), field=(1,2), dtype=np.double)
                dust_j  = hp.read_map('toast/output/Foregrounds/Gaussian/%04i/filterbin_coadd-hm1_map.fits'%(id_sim), field=(1,2), dtype=np.double)
            elif band_j == 'f090_hm2':
                cmb_j = hp.read_map('toast/output/Validation_for_paper/CMBl_pwf_beam/%04i/filterbin_coadd-hm2_map.fits'%(id_sim), field=(1,2), dtype=np.double)
                noise_j = hp.read_map('toast/output/Noise_for_paper/Atm_10m-reso/%04i/filterbin_coadd-hm2_map.fits'%(id_sim), field=(1,2), dtype=np.double)
                dust_j  = hp.read_map('toast/output/Foregrounds/Gaussian/%04i/filterbin_coadd-hm2_map.fits'%(id_sim), field=(1,2), dtype=np.double)
            elif band_j == 'f150_hm2':
                cmb_j = hp.read_map('toast/output/Validation_for_paper/CMBl_pwf_beam/%04i/filterbin_coadd-hm2_map.fits'%(id_sim), field=(1,2), dtype=np.double)
                noise_j = hp.read_map('toast/output/Noise_for_paper/Atm_10m-reso_f150/%04i/filterbin_coadd-hm2_map.fits'%(id_sim), field=(1,2), dtype=np.double)
                dust_j  = hp.read_map('toast/output/Foregrounds/Gaussian/%04i/filterbin_coadd-hm2_map.fits'%(id_sim), field=(1,2), dtype=np.double)
            elif band_j == 'f090_hm3':
                cmb_j = hp.read_map('toast/output/Validation_for_paper/CMBl_pwf_beam/%04i/filterbin_coadd-hm3_map.fits'%(id_sim), field=(1,2), dtype=np.double)
                noise_j = hp.read_map('toast/output/Noise_for_paper/Atm_10m-reso/%04i/filterbin_coadd-hm3_map.fits'%(id_sim), field=(1,2), dtype=np.double)
                dust_j  = hp.read_map('toast/output/Foregrounds/Gaussian/%04i/filterbin_coadd-hm3_map.fits'%(id_sim), field=(1,2), dtype=np.double)
            elif band_j == 'f150_hm3':
                cmb_j = hp.read_map('toast/output/Validation_for_paper/CMBl_pwf_beam/%04i/filterbin_coadd-hm3_map.fits'%(id_sim), field=(1,2), dtype=np.double)
                noise_j = hp.read_map('toast/output/Noise_for_paper/Atm_10m-reso_f150/%04i/filterbin_coadd-hm3_map.fits'%(id_sim), field=(1,2), dtype=np.double)
                dust_j  = hp.read_map('toast/output/Foregrounds/Gaussian/%04i/filterbin_coadd-hm3_map.fits'%(id_sim), field=(1,2), dtype=np.double)
            elif band_j == 'f090_hm4':
                cmb_j = hp.read_map('toast/output/Validation_for_paper/CMBl_pwf_beam/%04i/filterbin_coadd-hm4_map.fits'%(id_sim), field=(1,2), dtype=np.double)
                noise_j = hp.read_map('toast/output/Noise_for_paper/Atm_10m-reso/%04i/filterbin_coadd-hm4_map.fits'%(id_sim), field=(1,2), dtype=np.double)
                dust_j  = hp.read_map('toast/output/Foregrounds/Gaussian/%04i/filterbin_coadd-hm4_map.fits'%(id_sim), field=(1,2), dtype=np.double)
            elif band_j == 'f150_hm4':
                cmb_j = hp.read_map('toast/output/Validation_for_paper/CMBl_pwf_beam/%04i/filterbin_coadd-hm4_map.fits'%(id_sim), field=(1,2), dtype=np.double)
                noise_j = hp.read_map('toast/output/Noise_for_paper/Atm_10m-reso_f150/%04i/filterbin_coadd-hm4_map.fits'%(id_sim), field=(1,2), dtype=np.double)
                dust_j  = hp.read_map('toast/output/Foregrounds/Gaussian/%04i/filterbin_coadd-hm4_map.fits'%(id_sim), field=(1,2), dtype=np.double)
            elif band_j == '353_A_full':
                cmb_j = hp.read_map('toast/output/Validation_for_paper/Planck_CMB-noise_353A_pwf_beam/%04i/filterbin_coadd-full_map.fits'%(id_sim), field=(1,2), dtype=np.double)
                dust_j  = hp.read_map('toast/output/Foregrounds/Gaussian/%04i/filterbin_coadd-full_map.fits'%(id_sim),field=(1,2), dtype=np.double)
            elif band_j == '353_B_full':
                cmb_j = hp.read_map('toast/output/Validation_for_paper/Planck_CMB-noise_353B_pwf_beam/%04i/filterbin_coadd-full_map.fits'%(id_sim), field=(1,2), dtype=np.double)
                dust_j  = hp.read_map('toast/output/Foregrounds/Gaussian/%04i/filterbin_coadd-full_map.fits'%(id_sim),field=(1,2), dtype=np.double)
            
            sed_factor_j = sed_dust(freq_values[j], beta_dust, Tdust)
            if '353' in band_j:
                # The cmb already includes noise, I only need to correct the dust beam
                field_spin2_nobeam_j = nmt.NmtField(mask, [cmb_j[0], cmb_j[1]])
                field_spin2_beam_j = nmt.NmtField(mask, [dust_j[0]*sed_factor_j, dust_j[1]*sed_factor_j])
            else:
                # We are in f090 or f150, so cmb and dust have the same beam and I need to correct for it
                field_spin2_nobeam_j = nmt.NmtField(mask, [noise_j[0], noise_j[1]])
                field_spin2_beam_j = nmt.NmtField(mask, [cmb_j[0] + dust_j[0]*sed_factor_j, cmb_j[1] + dust_j[1]*sed_factor_j])
            beam_correction_j = (beams_array[j][:,1]/beams_array[0][:,1]) # this correction factor is needed for the cmb+dust
                        
            # cross spectra
            PCell_beami_beamj     = bins.bin_cell(nmt.compute_coupled_cell(field_spin2_beam_i, field_spin2_beam_j) * beam_correction_i * beam_correction_j )
            PCell_nobeami_beamj   = bins.bin_cell(nmt.compute_coupled_cell(field_spin2_nobeam_i, field_spin2_beam_j) * beam_correction_j )
            PCell_beami_nobeamj   = bins.bin_cell(nmt.compute_coupled_cell(field_spin2_beam_i, field_spin2_nobeam_j) * beam_correction_i )
            PCell_nobeami_nobeamj = bins.bin_cell(nmt.compute_coupled_cell(field_spin2_nobeam_i, field_spin2_nobeam_j) )
            PCell = PCell_beami_beamj + PCell_nobeami_beamj + PCell_beami_nobeamj + PCell_nobeami_nobeamj
            decoupled_pcl = inv_couplings[i,j] @ PCell.flatten()
            PCell_obs[id_sim_rel,i,j] = decoupled_pcl.reshape((size, Nbins))
    print('Done with validation %i in rank %i'%(id_sim,rank))

PCell_obs_total = None
if rank == 0:
    PCell_obs_total = np.empty((Nmc,)+PCell_obs.shape[1:], dtype=np.double)
comm.Gather(PCell_obs, PCell_obs_total, root=0)

if rank == 0:
    PCell_obs_total = np.moveaxis(PCell_obs_total,0,2)
    print('Shape of final array = ',PCell_obs_total.shape)
    np.save('output/Cells_FirstDayEveryMonth_Full_GaussianDust_nside512_fpthin8_CrossCL_pwf_beam', PCell_obs_total)