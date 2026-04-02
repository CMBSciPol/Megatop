import healpy as hp
import numpy as np
import pymaster as nmt

from megatop import DataManager
from megatop.utils import logger


def create_binning(nside, delta_ell, end_first_bin=None):
    """ """
    if end_first_bin is not None:
        bin_low = np.arange(end_first_bin, 3 * nside, delta_ell)
        bin_high = bin_low + delta_ell - 1
        bin_low = np.concatenate(([2], bin_low))
        bin_high = np.concatenate(([end_first_bin - 1], bin_high))
    else:
        bin_low = np.arange(0, 3 * nside, delta_ell)
        bin_high = bin_low + delta_ell - 1

        bin_low[0] = 2  # forcing to start at ell=2 without shifting all the other bins by 2
        # This must be done after defing bin_high to avoid overlap of zeroth and first bin

    bin_high[-1] = 3 * nside - 1
    bin_center = (bin_low + bin_high) / 2

    return bin_low, bin_high, bin_center


def load_nmt_binning(manager: DataManager):
    """Load the binning from the file."""
    binning_info = np.load(manager.path_to_binning, allow_pickle=True)
    # TODO: allow different binning functions?
    logger.info(f"Loading binning from {manager.path_to_binning}")
    return nmt.NmtBin.from_edges(binning_info["bin_low"], binning_info["bin_high"] + 1)


def compare_obsmat_vs_mask(obs_mat, binary_mask):
    list_diff_stokes = []
    for i in range(3):
        diag_obsmat_stokes = obs_mat.diagonal(0).reshape((3, hp.nside2npix(128)))[i]
        non_zero_diag_obsmat_stokes = diag_obsmat_stokes != 0
        non_zero_diag_obsmat_stokes_nest2ring = hp.reorder(non_zero_diag_obsmat_stokes, n2r=True)

        list_diff_stokes.append(np.sum(non_zero_diag_obsmat_stokes_nest2ring != binary_mask))

    list_diff_stokes = np.array(list_diff_stokes)
    bool_all_equal = np.all(list_diff_stokes == 0).astype(bool)
    if not bool_all_equal:
        logger.error(
            "The observation matrix and mask are not overlapping correctly, for each stokes params there are:",
            list_diff_stokes,
            " pixels not matching. THIS WILL LEAD TO SERIOUS ISSUE IN THE COMPONENT SEPARATION",
        )
    return bool_all_equal
