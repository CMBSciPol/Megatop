import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import IPython
from megatop.metadata_manager import BBmeta
import argparse
import os

def GetNoiseMapsMSS2(meta):
    maps_list = meta.maps_list
    print(maps_list)

    save_path = '/pscratch/sd/j/jost/SO_MEGATOP/MSS2/Coadd/NoiseMaps'

    for m in maps_list:
        print(m)
        # Load the map
        path_coadd = meta.get_map_filename(m, None)
        coadd_map = hp.read_map(path_coadd, field=None)
        
        path_fg = os.path.join(meta.map_directory, meta.fg_root_from_map_set(m)+'.fits')
        fg_map = hp.read_map(path_fg, field=None)

        path_cmb = os.path.join(meta.map_directory, meta.cmb_root_from_map_set(m)+'.fits')
        cmb_map = hp.read_map(path_cmb, field=None)

        noise_map = coadd_map - fg_map - cmb_map
        hp.write_map(os.path.join(save_path, meta.cmb_root_from_map_set(m).replace('cmb', 'noise')+'.fits'),
                      noise_map, overwrite=True)

    # 

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='simplistic simulator')
    parser.add_argument("--globals", type=str,
                        help="Path to yaml with global parameters")
    parser.add_argument("--use_mpi", action="store_true",
                        help="Use MPI instead of for loops to pre-process multiple maps, or simulate multiple sims.")
    parser.add_argument("--sims", default=None,
                        help="Generate a set of sims if True.")    
    parser.add_argument("--plots", action="store_true",
                        help="Plot the generated maps if True.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    meta = BBmeta(args.globals)

    GetNoiseMapsMSS2(meta)

