import pytest

from megatop.config import GeneralConfig, V3Noise, V3Sensitivity, yaml_converter


def test_lmax_validator():
    with pytest.raises(ValueError, match="less than or equal to"):
        GeneralConfig(nside=128, lmax=500)


@pytest.mark.parametrize("class_", [V3Sensitivity, V3Noise])
def test_structure_int_enums(class_) -> None:
    """Check that the converter un/structures our IntEnum subclasses by name."""
    for name, member in class_.__members__.items():
        assert yaml_converter.structure(name, class_) == member
        assert yaml_converter.unstructure(member) == name
