import argparse
from megatop.metadata_manager import BBmeta, Timer
from megatop import utils
from fgbuster.component_model import CMB, Dust, Synchrotron
import fgbuster as fg
import numpy as np
import os
import healpy as hp
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm
import time
import IPython
import pymaster as nmt


def compute_auto_cross_cl_from_maps_list(maps_dict, mask, beam, workspace, purify_e=True, purify_b=True, n_iter=3):

    # Create the fields
    fields = []
    for key in maps_dict.keys():
        fields.append(nmt.NmtField(mask, maps_dict[key], 
                                   beam=beam, 
                                   purify_e=purify_e,
                                   purify_b=purify_b,
                                   n_iter = n_iter))

    # Compute the power spectra
    cl_list = []
    for i, f_a in enumerate(fields):
        for j, f_b in enumerate(fields):
            if i <= j:
                cl_coupled = nmt.compute_coupled_cell(f_a, f_b)
                cl_decoupled = workspace.decouple_cell(cl_coupled)
                cl_list.append(cl_decoupled)

    # Store in dictionary with key x key
    cl_dict = {}
    for i, key in enumerate(maps_dict.keys()):
        for j, key2 in enumerate(maps_dict.keys()):
            if i <= j:
                cl_dict[key + 'x' + key2] = cl_list.pop(0)

    return cl_dict

def plot_all_Cls(all_Cls, bin_centre, file_path, cmb_theory_cls=None):

    # Define the labels
    labels = ['EE', 'EB', 'BE', 'BB']

    # Create the figure
    fig, axs = plt.subplots(4, 1, figsize=(10, 20))

    axs = axs.flatten()
 
    # Loop over the different power spectra
    for i, label in enumerate(labels):
        # Loop over the different components
        for j, key in enumerate(all_Cls.keys()):
            axs[i].plot(bin_centre, all_Cls[key][i], label=key, color='C' + str(j))
            axs[i].plot(bin_centre, -all_Cls[key][i], linestyle='--', color='C' + str(j))
        if cmb_theory_cls is not None:
            axs[i].plot(bin_centre, cmb_theory_cls[i], label='Input CMB', color='black')

        axs[i].set_title(label)
        axs[i].set_xlabel(r'$\ell$')
        axs[i].set_ylabel(r'$C_{\ell}$')
        axs[i].legend()
        axs[i].set_yscale('log')
        axs[i].set_xscale('log')

    plt.savefig(file_path)
    plt.close()

def plot_all_Cls_and_diffs(all_Cls, reference_cl, bin_centre, file_path, reference_name='Input CMB'):

    # Define the labels
    labels = ['EE', 'EB', 'BE', 'BB']

    # Create the figure
    fig, axs = plt.subplots(8, 1, figsize=(10, 20), gridspec_kw={'height_ratios': [2, 1, 2, 1, 2, 1, 2, 1]})
    axs = axs.flatten()
 
    # Loop over the different power spectra
    for i in range(axs.size):
        # Loop over the different components
        for j, key in enumerate(all_Cls.keys()):
            if not i % 2:
                axs[i].plot(bin_centre, all_Cls[key][int(i/2)], label=key, color='C' + str(j))
                axs[i].plot(bin_centre, -all_Cls[key][int(i/2)], linestyle='--', color='C' + str(j))
                axs[i].plot(bin_centre, reference_cl[int(i/2)], label=reference_name, color='black')
                
                axs[i].set_title(labels[int(i/2)])
                axs[i].set_xlabel(r'$\ell$')
                axs[i].set_ylabel(r'$C_{\ell}$')
                if i == 0:
                    axs[i].legend()

                axs[i].set_yscale('log')
                axs[i].set_xscale('log')

            else:
                if key=='CMBxCMB':
                    axs[i].plot(bin_centre, all_Cls[key][int(i/2)] - reference_cl[int(i/2)], label=key, color='C' + str(j))
                    xlims = axs[i].get_xlim()
                    axs[i].hlines(0, xlims[0], xlims[1], color='black', linestyle='--')
                    axs[i].set_xlim(xlims)
                    axs[i].set_xlabel(r'$\ell$')
                    axs[i].set_ylabel(r'$\Delta C_{\ell}$')
                    axs[i].set_xscale('log')

    plt.savefig(file_path)
    plt.close()




def spectra_estimation(args):
    meta = BBmeta(args.globals)
    timer_spectra = Timer()
    
    timer_spectra.start('loading_comp_maps')
    fname_comp_maps = os.path.join(meta.components_directory, 'components_maps.npy')
    comp_maps = np.load(fname_comp_maps)
    fname_invAtNA = os.path.join(meta.components_directory, 'invAtNA.npy')
    invAtNA = np.load(fname_invAtNA)
    timer_spectra.stop('loading_comp_maps', "Loading component maps", args.verbose)

    # Creating/loading bins
    
    bin_low, bin_high, bin_centre = utils.create_binning(meta.nside, meta.map2cl_pars['delta_ell'],
                                                         end_first_bin=meta.lmin)

    bin_index_lminlmax = np.where((bin_low >= meta.lmin) & (bin_high <= meta.lmax))[0]


    np.savez(os.path.join(meta.spectra_directory, 'binning.npz'),
             bin_low=bin_low, bin_high=bin_high, bin_centre=bin_centre, bin_index_lminlmax=bin_index_lminlmax)
    nmt_bins = nmt.NmtBin.from_edges(bin_low,
                                     bin_high + 1)    
    # b = nmt.NmtBin.from_nside_linear(meta.nside, nlb=int(meta.map2cl_pars['delta_ell']))
    
    # Loading analysis mask
    mask_analysis = meta.read_mask('analysis')

    # Initializin workspace
    timer_spectra.start('initializing_workspace')
    path_Cl_lens = meta.get_fname_cls_fiducial_cmb('lensed')
    Cl_lens = hp.read_cl(path_Cl_lens)

    wpix_out = hp.pixwin(meta.general_pars['nside'],pol=True,lmax=3*meta.nside) # Pixel window function of output maps
    Bl_gauss_common = hp.gauss_beam(np.radians(meta.pre_proc_pars['common_beam_correction']/60), lmax=3*meta.nside, pol=True)
    wpix_in = hp.pixwin( meta.general_pars["nside"],pol=True,lmax=3*meta.nside) # Pixel window function of input maps
    wpix_in[1][0:2] = 1. #in order not to divide by 0

    # IPython.embed()
    effective_beam = Bl_gauss_common[:,1] * wpix_out[1] #/ wpix_in[1]
    map_T_init_wsp, map_Q_init_wsp, map_U_init_wsp = hp.synfast(Cl_lens, meta.nside, new=True)
    fields_init_wsp = nmt.NmtField(mask_analysis, [map_Q_init_wsp, map_U_init_wsp], 
                                   beam=effective_beam,
                                   purify_e=meta.map2cl_pars['purify_e'],
                                   purify_b=meta.map2cl_pars['purify_b'],
                                   n_iter = meta.map2cl_pars['n_iter_namaster'])
    workspace_cc = nmt.NmtWorkspace()
    workspace_cc.compute_coupling_matrix(fields_init_wsp, fields_init_wsp, nmt_bins,
                                         n_iter=meta.map2cl_pars['n_iter_namaster'])
    timer_spectra.stop('initializing_workspace', "Initializing workspace", args.verbose)
    # TODO: update namaster and test from_fields for workspace (see SOOPERCOOL)
    # workspaceff = nmt.NmtWorkspace.from_fields(fields_init_wsp,fields_init_wsp,nmt_bins)

    # Testing the function
    timer_spectra.start('spectra_estimation')
    comp_dict = {'CMB': comp_maps[0], 'Dust': comp_maps[1], 'Synch': comp_maps[2]}
    all_Cls = compute_auto_cross_cl_from_maps_list(comp_dict, mask_analysis, effective_beam, workspace_cc, purify_e=meta.map2cl_pars['purify_e'], 
                                                   purify_b=meta.map2cl_pars['purify_b'], n_iter=meta.map2cl_pars['n_iter_namaster'])    
    
    np.savez(os.path.join(meta.spectra_directory, 'cross_components_Cls.npz'), **all_Cls)

    if args.plots: all_Clslminlmax = utils.apply_lminlmax_to_dict(all_Cls, bin_index_lminlmax)
    timer_spectra.stop('spectra_estimation', "Spectra estimation", args.verbose)


    timer_spectra.start('noise_spectra_estimation')
    noise_dict = {'NoiseCMB': invAtNA[0,0], 'NoiseDust': invAtNA[1,1], 'NoiseSynch': invAtNA[2,2]}
    noise_dict_offdiag = {'NoiseCMBDust': invAtNA[0,1], 'NoiseDustSynch': invAtNA[1,2], 'NoiseCMBSynch': invAtNA[0,2]} 
    # Here we assume that InvAtNA is symmetric, which seems true up to numerical precision
    Cls_noise = compute_auto_cross_cl_from_maps_list(noise_dict, mask_analysis, effective_beam, workspace_cc, purify_e=meta.map2cl_pars['purify_e'], 
                                                   purify_b=meta.map2cl_pars['purify_b'], n_iter=meta.map2cl_pars['n_iter_namaster'])
    np.savez(os.path.join(meta.spectra_directory, 'noise_Cls.npz'), **Cls_noise)    

    if args.plots: Cls_noiselminlmax = utils.apply_lminlmax_to_dict(Cls_noise, bin_index_lminlmax)
    

    Cls_noise_offdiag = compute_auto_cross_cl_from_maps_list(noise_dict_offdiag, mask_analysis, effective_beam, workspace_cc, purify_e=meta.map2cl_pars['purify_e'],
                                                    purify_b=meta.map2cl_pars['purify_b'], n_iter=meta.map2cl_pars['n_iter_namaster'])
    np.savez(os.path.join(meta.spectra_directory, 'noise_Cls_offdiag.npz'), **Cls_noise_offdiag)
    
    if args.plots: Cls_noise_offdiaglminlmax = utils.apply_lminlmax_to_dict(Cls_noise_offdiag, bin_index_lminlmax)
    
    timer_spectra.stop('noise_spectra_estimation', "Noise spectra estimation", args.verbose)
    
    if args.plots:
        timer_spectra.start('plotting')
        plot_dir = meta.plot_dir_from_output_dir(meta.spectra_directory_rel)
        input_cmb_spectra = utils.get_Cl_CMB_model_from_meta(meta)[0][:,:3*meta.nside]
        binned_input_cmb_spectra = nmt_bins.bin_cell(input_cmb_spectra)
        reshape_input_cmb_spectra = np.array([binned_input_cmb_spectra[1], binned_input_cmb_spectra[-2], binned_input_cmb_spectra[-2], binned_input_cmb_spectra[2]])

        plot_all_Cls(all_Clslminlmax, bin_centre[bin_index_lminlmax], plot_dir+'/all_Cls.png', 
                    cmb_theory_cls=reshape_input_cmb_spectra[:,bin_index_lminlmax])
        plot_all_Cls_and_diffs({'CMBxCMB':all_Clslminlmax['CMBxCMB']}, reshape_input_cmb_spectra[:,bin_index_lminlmax], bin_centre[bin_index_lminlmax], plot_dir+'/Delta_CMB.png', reference_name='Input CMB')

        unbiased_Cls = {}
        for key_signal, key_noise in zip(all_Cls.keys(), Cls_noise.keys()):
            unbiased_Cls[key_signal] = all_Clslminlmax[key_signal] - Cls_noiselminlmax[key_noise]
        plot_all_Cls(unbiased_Cls, bin_centre[bin_index_lminlmax],plot_dir+'/unbiased_Cls.png',
                    cmb_theory_cls=reshape_input_cmb_spectra[:,bin_index_lminlmax])
        plot_all_Cls_and_diffs({'CMBxCMB':unbiased_Cls['CMBxCMB']}, reshape_input_cmb_spectra[:,bin_index_lminlmax], bin_centre[bin_index_lminlmax], plot_dir+'/Delta_CMB_debiased.png', reference_name='Input CMB')
        
        plot_all_Cls(Cls_noiselminlmax, bin_centre[bin_index_lminlmax], plot_dir+'/Noise_Cls.png', cmb_theory_cls=reshape_input_cmb_spectra[:,bin_index_lminlmax])
        plot_all_Cls(Cls_noise_offdiaglminlmax, bin_centre[bin_index_lminlmax], plot_dir+'/Noise_Cls_offdiag.png', 
                    cmb_theory_cls=reshape_input_cmb_spectra[:,bin_index_lminlmax])
        timer_spectra.stop('plotting', "Plotting", args.verbose)

    return all_Cls






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='simplistic simulator') #TODO change name ??
    parser.add_argument("--globals", type=str,
                        help="Path to yaml with global parameters")
    parser.add_argument("--plots", action="store_true",
                        help="Plot the generated maps if True.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    res = spectra_estimation(args)
    