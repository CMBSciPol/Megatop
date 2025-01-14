import argparse
from megatop.metadata_manager import BBmeta, Timer
import healpy as hp
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import megatop.V3calc as V3
from megatop import utils
import pymaster as nmt
import IPython

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

    # Set up figure and gridspec
    fig = plt.figure(figsize=(6, 16))
    # 8 rows total: 2 for each plot pair + 2 for spacing
    gs = gridspec.GridSpec(11, 1, figure=fig, height_ratios=[1, 1, 0.6, 1, 1, 0.6, 1, 1, 0.6, 1, 1])

    # Plot each pair with shared x-axis and no space between
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[3])
    ax4 = fig.add_subplot(gs[4], sharex=ax3)
    ax5 = fig.add_subplot(gs[6])
    ax6 = fig.add_subplot(gs[7], sharex=ax5)
    ax7 = fig.add_subplot(gs[9])
    ax8 = fig.add_subplot(gs[10], sharex=ax7)

    # Hide x-axis labels for the top plots of each pair
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax3.get_xticklabels(), visible=False)
    plt.setp(ax5.get_xticklabels(), visible=False)

    # Reduce spacing between plots
    gs.update(hspace=0)  # No space between subplots within pairs

    # Add spacing between the pairs
    fig.subplots_adjust(hspace=0.5)  # Space only between pairs

    for i, (top_ax, bot_ax) in enumerate(zip( [ax1, ax3, ax5, ax7], [ax2, ax4, ax6, ax8])):

        for j, key in enumerate(all_Cls.keys()):   
            
            top_ax.plot(bin_centre, all_Cls[key][i], label=key, color='C' + str(j))
            top_ax.plot(bin_centre, -all_Cls[key][i], linestyle='--', color='C' + str(j))
            top_ax.plot(bin_centre, reference_cl[i], label=reference_name, color='black')
            
            top_ax.set_title(labels[i])
            top_ax.set_xlabel(r'$\ell$')
            top_ax.set_ylabel(r'$C_{\ell}$')
            if i == 0:
                top_ax.legend()

            top_ax.set_yscale('log')
            top_ax.set_xscale('log')

            if key=='CMBxCMB':
                bot_ax.plot(bin_centre, all_Cls[key][i] - reference_cl[i], label=key, color='C' + str(j))
                xlims = bot_ax.get_xlim()
                bot_ax.hlines(0, xlims[0], xlims[1], color='black', linestyle='--')
                bot_ax.set_xlim(xlims)
                bot_ax.set_xlabel(r'$\ell$')
                bot_ax.set_ylabel(r'$\Delta C_{\ell}$')
                bot_ax.set_xscale('log')

    plt.savefig(file_path)
    plt.close()    



def main_spectra_plotting(meta):
    mask = meta.read_mask('binary').astype(bool)
    fsky_mask = 1 # sum(mask) / len(mask) 

    bin_low, bin_high, bin_centre = utils.create_binning(meta.nside, meta.map2cl_pars['delta_ell'],
                                                         end_first_bin=meta.lmin)
    bin_index_lminlmax = np.where((bin_low >= meta.plot_pars['lmin_plot']) & (bin_high <= meta.plot_pars['lmax_plot']))[0]
    nmt_bins = nmt.NmtBin.from_edges(bin_low,
                                     bin_high + 1)    

    all_Cls = np.load(os.path.join(meta.spectra_directory, 'cross_components_Cls.npz'), allow_pickle=True)
    all_Clslminlmax = utils.apply_lminlmax_to_dict(all_Cls, bin_index_lminlmax) 
    for key in all_Clslminlmax.keys():
        all_Clslminlmax[key] = all_Clslminlmax[key] * fsky_mask
    
    Cls_noise = np.load(os.path.join(meta.spectra_directory, 'noise_Cls.npz'), allow_pickle=True)  
    Cls_noiselminlmax = utils.apply_lminlmax_to_dict(Cls_noise, bin_index_lminlmax) 
    for key in Cls_noiselminlmax.keys():
        Cls_noiselminlmax[key] = Cls_noiselminlmax[key] * fsky_mask
    # IPython.embed()
    # for key in Cls_noiselminlmax.keys():
    #     Cls_noiselminlmax[key] = np.sqrt(Cls_noiselminlmax[key])
    

    Cls_noise_offdiag = np.load(os.path.join(meta.spectra_directory, 'noise_Cls_offdiag.npz'), allow_pickle=True)    
    Cls_noise_offdiaglminlmax = utils.apply_lminlmax_to_dict(Cls_noise_offdiag, bin_index_lminlmax)
    for key in Cls_noise_offdiaglminlmax.keys():
        Cls_noise_offdiaglminlmax[key] = Cls_noise_offdiaglminlmax[key] * fsky_mask 

    plot_dir = meta.plot_dir_from_output_dir(meta.spectra_directory_rel)

    input_cmb_spectra = utils.get_Cl_CMB_model_from_meta(meta)[0][:,:3*meta.nside]
    binned_input_cmb_spectra = nmt_bins.bin_cell(input_cmb_spectra)
    reshape_input_cmb_spectra = np.array([binned_input_cmb_spectra[1], binned_input_cmb_spectra[-2], binned_input_cmb_spectra[-2], binned_input_cmb_spectra[2]])

    plot_all_Cls(all_Clslminlmax, bin_centre[bin_index_lminlmax], plot_dir+'/all_Cls.png', 
                cmb_theory_cls=reshape_input_cmb_spectra[:,bin_index_lminlmax])
    plot_all_Cls_and_diffs({'CMBxCMB':all_Clslminlmax['CMBxCMB']}, reshape_input_cmb_spectra[:,bin_index_lminlmax], 
                            bin_centre[bin_index_lminlmax], plot_dir+'/Delta_CMB.png', reference_name='Input CMB')

    unbiased_Cls = {}
    for key_signal, key_noise in zip(all_Cls.keys(), Cls_noise.keys()):
        unbiased_Cls[key_signal] = all_Clslminlmax[key_signal] - Cls_noiselminlmax[key_noise]
    plot_all_Cls(unbiased_Cls, bin_centre[bin_index_lminlmax],plot_dir+'/unbiased_Cls.png',
                cmb_theory_cls=reshape_input_cmb_spectra[:,bin_index_lminlmax])
    plot_all_Cls_and_diffs({'CMBxCMB':unbiased_Cls['CMBxCMB']}, reshape_input_cmb_spectra[:,bin_index_lminlmax], 
                            bin_centre[bin_index_lminlmax], plot_dir+'/Delta_CMB_debiased.png', 
                            reference_name='Input CMB')
    
    plot_all_Cls(Cls_noiselminlmax, bin_centre[bin_index_lminlmax], plot_dir+'/Noise_Cls.png', 
                    cmb_theory_cls=reshape_input_cmb_spectra[:,bin_index_lminlmax])
    plot_all_Cls(Cls_noise_offdiaglminlmax, bin_centre[bin_index_lminlmax], plot_dir+'/Noise_Cls_offdiag.png', 
                cmb_theory_cls=reshape_input_cmb_spectra[:,bin_index_lminlmax])
    IPython.embed()

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='simplistic simulator') #TODO change name ??
    parser.add_argument("--globals", type=str,
                        help="Path to yaml with global parameters")
    parser.add_argument("--plots", action="store_true",
                        help="Plot the generated maps if True.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if not args.plots:
        exit()

    timer_plots = Timer()
    timer_plots.start('plots_spectra')

    print('\n\nPlotting Map to Cls outputs\n\n')

    meta = BBmeta(args.globals)
    
    main_spectra_plotting(meta)

    timer_plots.stop('plots_spectra', "Maps to Cl outputs plots", args.verbose)

    print('\n\nPlotting Map to Cl outputs completed succesfully\n\n')
    