import numpy as np
from astropy.table import QTable
from scipy import constants

from ..config import Config
from ..data_manager import DataManager
from .logger import logger


def _cmb2bb(nu):
    T_CMB = 2.72548
    x = nu * constants.h * 1e9 / constants.k / T_CMB
    return np.exp(x) * (nu * x / np.expm1(x)) ** 2


def passband_constructor(config: Config, manager: DataManager, passband_int: bool) -> list:
    """Read passbands from config file and outputs"""  # TODO doc and shape/type of output
    for map_set in config.map_sets:
        if passband_int:
            tab = QTable.read(
                manager.path_to_passbands / map_set.passband_filename, format="ascii.ipac"
            )
            map_set.frequency = tab["bandpass_frequency"]
            map_set.weight = tab["bandpass_weights"]
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
        trans_norm = np.trapz(map_set.weight * _cmb2bb(map_set.frequency), map_set.frequency)  # noqa: NPY201
        trans = map_set.weight / trans_norm * _cmb2bb(map_set.frequency)
        passbands_norm.append([map_set.frequency, trans])
    return passbands_norm
