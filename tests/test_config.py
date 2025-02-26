import pytest

from megatop.config import (
    Config,
    GeneralConfig,
    MapSetConfig,
    V3Noise,
    V3Sensitivity,
    yaml_converter,
)


@pytest.fixture
def example_config():
    return Config.get_example()


def test_lmax_validator():
    """Check that config instantiation fails when lmax is too high."""
    with pytest.raises(ValueError, match="less than or equal to"):
        GeneralConfig(nside=128, lmax=500)


def test_map_set_name():
    """Check that the map set name is set and correctly formatted."""
    map_set = MapSetConfig(freq_tag=27, exp_tag="SAT4")
    assert hasattr(map_set, "name")
    assert map_set.name == "SAT4_f027"


@pytest.mark.parametrize("class_", [V3Sensitivity, V3Noise])
def test_structure_int_enums(class_) -> None:
    """Check that the converter un/structures our IntEnum subclasses by name."""
    for name, member in class_.__members__.items():
        assert yaml_converter.structure(name, class_) == member
        assert yaml_converter.unstructure(member) == name


def test_split_map_sets_one_group(example_config: Config) -> None:
    assert example_config.split_map_sets(1).map_sets == example_config.map_sets


@pytest.mark.parametrize("n_groups", [1, 2, 3, 4])
def test_split_map_sets_more_groups(n_groups: int, example_config: Config) -> None:
    num_sets = len(example_config.map_sets)

    def sconf(c: int):
        return example_config.split_map_sets(n_groups, color=c)

    total_sets = sum(len(sconf(c).map_sets) for c in range(n_groups))
    assert total_sets == num_sets
