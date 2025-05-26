import numpy as np
from astropy.table import QTable
from fgbuster.observation_helpers import _jysr2rj, _rj2cmb

from ..config import Config
from ..data_manager import DataManager
from .logger import logger

try:
    from numpy import trapezoid
except ImportError:
    from numpy import trapz as trapezoid

# def _cmb2bb(nu):
#     T_CMB = 2.72548
#     x = nu * constants.h * 1e9 / constants.k / T_CMB
#     return np.exp(x) * (nu * x / np.expm1(x)) ** 2


def passband_constructor(config: Config, manager: DataManager, passband_int: bool) -> list:
    """Read passbands from config file and outputs"""  # TODO doc and shape/type of output
    for map_set in config.map_sets:
        if passband_int:
            tab = QTable.read(
                manager.path_to_passbands / map_set.passband_filename, format="ascii.ipac"
            )
            map_set.frequency = np.array(tab["bandpass_frequency"])
            map_set.weight = np.array(tab["bandpass_weight"])
        else:
            map_set.frequency = map_set.freq_tag
            map_set.weight = 1.0
    return config.map_sets


def standardize_passbands(passbands: dict):
    logger.warning("NO CHECKS PERFORMED FOR NOW !! BE CAREFUL WITH PASSBAND INTEGRATION")
    return passbands


def fgbuster_passband(map_sets: list) -> list:
    """"""
    passbands_norm = []
    for map_set in map_sets:
        weight = map_set.weight / _jysr2rj(map_set.frequency)
        weight /= _rj2cmb(
            map_set.frequency
        )  # so now on top of the weight we have the conversion factor from K_CMB to Jy/sr
        weight /= trapezoid(np.nan_to_num(weight, nan=0), map_set.frequency * 1e9)
        passbands_norm.append([map_set.frequency, np.nan_to_num(weight, nan=0)])
    return passbands_norm
