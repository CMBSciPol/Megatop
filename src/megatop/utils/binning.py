import numpy as np
import pymaster as nmt

from megatop import DataManager


def create_binning(
    lmax: int, delta_ell: int, uniform_start: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create multipole bin edges for NaMaster.

    All edges are inclusive: bin i covers ell in [bin_low[i], bin_high[i]].
    The first bin always starts at ell=2 and the last bin always ends at lmax.
    When saving and reloading via NmtBin.from_edges, pass bin_high + 1
    because namaster treats its upper edge argument as exclusive.

    Args:
        lmax: Maximum multipole, inclusive. The last bin always ends at lmax.
        delta_ell: Width of the uniform bins.
        uniform_start: If provided, the first bin spans [2, uniform_start - 1]
            and uniform bins of width delta_ell start at uniform_start.
            If None, all bins have width delta_ell aligned to multiples of
            delta_ell from 0, with the first bin forced to start at ell=2.

    Returns:
        bin_low: Lower edge of each bin (inclusive).
        bin_high: Upper edge of each bin (inclusive).
        bin_center: Arithmetic center of each bin.
    """
    if uniform_start is not None:
        # Uniform bins of width delta_ell starting at uniform_start
        regular_low = np.arange(uniform_start, lmax, delta_ell)
        regular_high = regular_low + delta_ell - 1
        # Prepend the wide first bin [2, uniform_start - 1]
        bin_low = np.concatenate(([2], regular_low))
        bin_high = np.concatenate(([uniform_start - 1], regular_high))
    else:
        # Uniform grid aligned to multiples of delta_ell from 0
        bin_low = np.arange(0, lmax, delta_ell)
        bin_high = bin_low + delta_ell - 1
        # Override first bin to start at ell=2 without shifting the whole grid.
        # Must be done after computing bin_high to avoid overlap with bin 1.
        bin_low[0] = 2

    bin_high[-1] = lmax  # clamp last bin to lmax (inclusive)
    bin_center = (bin_low + bin_high) / 2

    return bin_low, bin_high, bin_center


def load_nmt_binning(manager: DataManager):
    """Load the binning from the file."""
    binning_info = np.load(manager.path_to_binning, allow_pickle=True)
    # +1 because NaMaster expects upper bounds to be exclusive
    return nmt.NmtBin.from_edges(binning_info["bin_low"], binning_info["bin_high"] + 1)