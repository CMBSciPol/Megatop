from enum import IntEnum

from cattrs.preconf.pyyaml import make_converter

__all__ = [
    "yaml_converter",
]


def _is_intenum(cls):
    return issubclass(cls, IntEnum)


# YAML converter that catches typos
yaml_converter = make_converter(forbid_extra_keys=True)

# unstructure IntEnum subclasses by name instead of value
yaml_converter.register_unstructure_hook_func(_is_intenum, lambda val: val.name)

# TODO: that one doesn't work -- it's not called
# yaml_converter.register_structure_hook_func(_is_intenum, lambda val, cls: cls[val])
