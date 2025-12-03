from pdoc import __pdoc__

from . import utils
from .config import Config
from .data_manager import DataManager

__all__ = [
    "Config",
    "DataManager",
    "utils",
]

# disable pdoc for selected modules
__pdoc__["megatop.pipeline.TF_computation_interface"] = False
