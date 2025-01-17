import argparse

import healpy as hp
import numpy as np

from megatop.utils.metadata_manager import BBmeta
from megatop.utils.preproc_utils import (
    CommonBeamConvAndNsideModification,
    read_maps,
    read_maps_ben_sims,
    save_preprocessed_maps,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="simplistic simulator")  # TODO change name ?
    parser.add_argument("--globals", type=str, help="Path to yaml with global parameters")

    args = parser.parse_args()
    meta = BBmeta(args.globals)

    if hasattr(meta, "ben_sims") and meta.ben_sims:
        meta.logger.info("Reading ben sims")
        meta.logger.info('WARNING: ben_sims related function will be removed')
        # TODO: remove ben_sims cases
        input_maps = read_maps_ben_sims(meta, id_sim=meta.id_sim)
    else:
        input_maps = read_maps(meta)
    
    meta.logger.info("input_maps shape = {}".format( input_maps.shape))

    if np.all(
        np.array(meta.pre_proc_pars["common_beam_correction"])
        == np.array(meta.pre_proc_pars["fwhm"])
    ):
        meta.logger.info("Common beam correction is the same as the input beam, no need to apply it.")
        meta.logger.info("WARNING: this is mostly for testing it might not actually represent the real noise")
        convolved_maps = input_maps.astype("float64")

    else:
        convolved_maps = CommonBeamConvAndNsideModification(meta, input_maps)

    # Applying binary mask
    binary_mask_path = meta.get_fname_mask("binary")
    binary_mask = hp.read_map(binary_mask_path, dtype=float)
    masked_convolved_maps = convolved_maps * binary_mask
    
    save_preprocessed_maps(meta, masked_convolved_maps)

    meta.logger.info("\n\nPre-processing step completed succesfully\n\n")


