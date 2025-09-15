import numpy as np
import pymaster as nmt

from megatop import DataManager
from megatop.utils import logger


def create_binning(nside, delta_ell, end_first_bin=None):
    """ """
    if end_first_bin is not None:
        bin_low = np.arange(end_first_bin, 3 * nside, delta_ell)
        bin_high = bin_low + delta_ell - 1
        bin_low = np.concatenate(([0], bin_low))
        bin_high = np.concatenate(([end_first_bin - 1], bin_high))
    else:
        bin_low = np.arange(0, 3 * nside, delta_ell)
        bin_high = bin_low + delta_ell - 1
    bin_high[-1] = 3 * nside - 1
    bin_center = (bin_low + bin_high) / 2

    return bin_low, bin_high, bin_center


def load_nmt_binning(manager: DataManager):
    """Load the binning from the file."""
    binning_info = np.load(manager.path_to_binning, allow_pickle=True)
    # TODO: allow different binning functions?
    logger.info(f"Loading binning from {manager.path_to_binning}")
    return nmt.NmtBin.from_edges(binning_info["bin_low"], binning_info["bin_high"] + 1)
