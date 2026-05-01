import numpy as np
import pymaster as nmt

from megatop import DataManager
from megatop.utils import logger


def create_binning(lmax, delta_ell, end_first_bin=None):
    """ """
    if end_first_bin is not None:
        bin_low = np.arange(end_first_bin, lmax, delta_ell)
        bin_high = bin_low + delta_ell - 1
        bin_low = np.concatenate(([2], bin_low))
        bin_high = np.concatenate(([end_first_bin - 1], bin_high))
    else:
        bin_low = np.arange(0, lmax, delta_ell)
        bin_high = bin_low + delta_ell - 1

        bin_low[0] = 2  # forcing to start at ell=2 without shifting all the other bins by 2
        # This must be done after defing bin_high to avoid overlap of zeroth and first bin

    bin_high[-1] = lmax
    bin_center = (bin_low + bin_high) / 2

    return bin_low, bin_high, bin_center


def load_nmt_binning(manager: DataManager):
    """Load the binning from the file."""
    binning_info = np.load(manager.path_to_binning, allow_pickle=True)
    # TODO: allow different binning functions?
    logger.info(f"Loading binning from {manager.path_to_binning}")
    return nmt.NmtBin.from_edges(binning_info["bin_low"], binning_info["bin_high"] + 1)
