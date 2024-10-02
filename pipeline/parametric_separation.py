import argparse
from megatop.metadata_manager import BBmeta, Timer
from fgbuster.component_model import CMB, Dust, Synchrotron
import fgbuster as fg
import numpy as np
import os
import healpy as hp
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm
import time


def weighted_comp_sep(args):
    meta = BBmeta(args.globals)
    timer_compsep = Timer()
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

    #TODO : read methods and options from file
    options={'disp':args.verbose, 'gtol': 1e-12, 'eps': 1e-12, 'maxiter': 100, 'ftol': 1e-12 } 
    tol=1e-18
    method='TNC'

    res = fg.separation_recipes.weighted_comp_sep(components, instrument,
                                                  data=freq_maps_preprocessed[:,1:], 
                                                  cov=noise_cov[:,1:], # Slice to remove the T maps, otherwise the separation will be biased
                                                  options=options, tol=tol, method=method)
    
    if args.verbose: print('success: ',res.success)
    timer_compsep.start('compsep')
    print(res.s.shape)
    return res
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='simplistic simulator') #TODO change name ??
    parser.add_argument("--globals", type=str,
                        help="Path to yaml with global parameters")
    parser.add_argument("--plots", action="store_true",
                        help="Plot the generated maps if True.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    res = weighted_comp_sep(args)