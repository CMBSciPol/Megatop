import argparse
from megatop.metadata_manager import BBmeta, Timer
from fgbuster.component_model import CMB, Dust, Synchrotron
from fgbuster.mixingmatrix import MixingMatrix
import fgbuster as fg
import numpy as np
import os
import healpy as hp
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm
import time
import IPython

def weighted_comp_sep(args):
    meta = BBmeta(args.globals)
    timer_compsep = Timer()
    timer_compsep.start('full_step')
    timer_compsep.start('loading_covmat')

    fname_covmat = os.path.join(meta.covmat_directory, 'pixel_noise_cov_preprocessed.npy')
    if args.verbose: print(f"Loading noise covariance from {fname_covmat}")
    noise_cov = np.load(fname_covmat)
    timer_compsep.stop('loading_covmat', "Loading noise covariance", args.verbose)

    timer_compsep.start('loading_maps')
    fname_preproc_maps = os.path.join(meta.pre_process_directory, 'freq_maps_preprocessed.npy')
    if args.verbose: print(f"Loading pre-processed frequency maps from {fname_preproc_maps} ")
    freq_maps_preprocessed = np.load(fname_preproc_maps)

    timer_compsep.stop('loading_maps', "Loading pre-processed frequency maps", args.verbose)

    timer_compsep.start('compsep')
    instrument = {'frequency': meta.frequencies}
    components = [CMB(), Dust(150., temp=20.0), Synchrotron(150.)]

    options = meta.parametric_sep_pars['options']
    tol = meta.parametric_sep_pars['tol']
    method = meta.parametric_sep_pars['method']

    res = fg.separation_recipes.weighted_comp_sep(components, instrument,
                                                  data=freq_maps_preprocessed[:,1:], 
                                                  cov=noise_cov[:,1:], # Slice to remove the T maps, otherwise the separation will be biased
                                                  options=options, tol=tol, method=method)
    
    if args.verbose: print('success: ',res.success)
    if args.verbose: print('results: ',res.x)
    timer_compsep.stop('compsep', "Component separation", args.verbose)
    
    A = MixingMatrix(*components)
    A_ev = A.evaluator(np.array(instrument['frequency']))
    A_maxL = A_ev(res.x)
    res.A_maxL = A_maxL

    # IPython.embed()
    # test_invAtNA = np.linalg.inv(np.einsum('cf,fqp,fs->csqp', A_maxL.T, 1/noise_cov[:,1:], A_maxL).T).T
    # sanity_check = np.max(np.abs((test_invAtNA - res.invAtNA) / res.invAtNA * 100))

    # test_invAtNA_U = np.dot(A_maxL.T, np.dot(1/noise_cov[:,2], A_maxL))
    # sanity_check = np.linalg.inv(A_maxL.T @ noise_cov @ A_maxL) - res.invAtNA
    # W_maxL = res.invAtNA @ A_maxL.T @ np.linalg.inv(noise_cov)

    res_dict = {}
    for attr in dir(res):
        if not attr.startswith('__'):
            res_dict[attr] = getattr(res, attr)
    np.savez(os.path.join(meta.components_directory, 'comp_sep_results.npz'), **res_dict)

    # res.s and res.invAtNA are saved twice, but they are the direct needed outputs for the next step
    # space could be saved by adding an if statement in the above dict construction (TODO?)
    np.save(os.path.join(meta.components_directory, 'components_maps.npy'), res.s)
    np.save(os.path.join(meta.components_directory, 'invAtNA.npy'), res.invAtNA)

    if args.plots:
        timer_compsep.start('plotting')
        components_results_plotting(res, meta)
        timer_compsep.stop('plotting', "Plotting", args.verbose)

    timer_compsep.stop('full_step', "Full component separation step", args.verbose)
    return res
    
def components_results_plotting(res, meta):
    binary_mask = meta.read_mask('binary').astype(bool)
    res.s[..., np.where(binary_mask==0)[0]] = hp.UNSEEN	

    plot_dir = meta.plot_dir_from_output_dir(meta.components_directory_rel)
    
    fig = plt.figure(figsize=(12, 12))
    for i, component_label in enumerate(['CMB', 'Dust', 'Synchrotron']):
        for j, stokes_label in enumerate(['Q', 'U']):
            hp.mollview(res.s[i,j], title= component_label + ' ' + stokes_label, 
                        sub=(3, 2, (2*i+j)+1), fig=fig, cbar=True)
    plt.savefig(plot_dir+'/components_maps.png')
    plt.close()

    res.invAtNA[..., np.where(binary_mask==0)[0]] = hp.UNSEEN	

    fig = plt.figure(figsize=(12, 12))
    for i, component_label in enumerate(['Noise CMB', 'Noise Dust', 'Noise Synchrotron']):
        for j, stokes_label in enumerate(['Q', 'U']):
            hp.mollview(res.invAtNA[i,i,j], title= component_label + '--' + stokes_label + ' -- norm = log' , 
                        sub=(3, 2, (2*i+j)+1), fig=fig, cbar=True, norm='log')
    plt.savefig(plot_dir+'/noise_per_components_maps.png')
    plt.close()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='simplistic simulator') #TODO change name ??
    parser.add_argument("--globals", type=str,
                        help="Path to yaml with global parameters")
    parser.add_argument("--plots", action="store_true",
                        help="Plot the generated maps if True.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    res = weighted_comp_sep(args)
    