from enum import IntEnum
from typing import Union, get_args, get_origin
from cattrs.preconf.pyyaml import make_converter

__all__ = [
    "yaml_converter",
]

def _is_intenum(cls):
    try:
        return issubclass(cls, IntEnum)
    except TypeError:
        return False

def _is_optional_float(cls):
    return (
        get_origin(cls) is Union
        and float in get_args(cls)
        and type(None) in get_args(cls)
    )

yaml_converter = make_converter(forbid_extra_keys=True)

yaml_converter.register_unstructure_hook_func(_is_intenum, lambda val: val.name)

yaml_converter.register_structure_hook_func(
    _is_optional_float,
    lambda val, _: None if (val is None or val == 'None') else float(val)
)

# Override le hook pour float pour gérer les strings 'None'
yaml_converter.register_structure_hook(
    float,
    lambda val, _: None if (val is None or val == 'None') else float(val)
)