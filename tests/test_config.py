from pathlib import Path

import pytest

from megatop import DataManager
from megatop.config import (
    Config,
    DataDirsConfig,
    FiducialCMBConfig,
    GeneralConfig,
    MapSetConfig,
    MapSimConfig,
    OutputDirsConfig,
    V3Noise,
    V3Sensitivity,
    _CAMBCosmoPars,
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
    map_set = MapSetConfig(freq_tag=27, exp_tag="SAT4", nhits_map_path="SO_nominal", beam=30)
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

    sub_configs = [sconf(c) for c in range(n_groups)]

    # check that the total number of map sets is conserved
    total_sets = sum(len(sc.map_sets) for sc in sub_configs)
    assert total_sets == num_sets


@pytest.fixture
def minimal_config():
    return Config(
        data_dirs=DataDirsConfig(root="data"),
        output_dirs=OutputDirsConfig(root="out"),
        fiducial_cmb=FiducialCMBConfig(compute_from_camb=True, camb_cosmo_pars=_CAMBCosmoPars()),
        map_sets=[MapSetConfig(freq_tag=93, exp_tag="SO", beam=30.0, nhits_map_path="SO_nominal")],
    )


def test_filter_sims_requires_obsmat_path(minimal_config: Config) -> None:
    """Config with filter_sims=True must have obsmat_path set on all map sets."""
    with pytest.raises(ValueError, match="obsmat_path"):
        minimal_config.map_sim_pars = MapSimConfig(filter_sims=True)
        Config(
            data_dirs=minimal_config.data_dirs,
            output_dirs=minimal_config.output_dirs,
            fiducial_cmb=minimal_config.fiducial_cmb,
            map_sets=minimal_config.map_sets,
            map_sim_pars=MapSimConfig(filter_sims=True),
        )


def test_filter_sims_with_obsmat_path_passes() -> None:
    """Config with filter_sims=True and obsmat_path set should not raise."""
    Config(
        data_dirs=DataDirsConfig(root="data"),
        output_dirs=OutputDirsConfig(root="out"),
        fiducial_cmb=FiducialCMBConfig(compute_from_camb=True, camb_cosmo_pars=_CAMBCosmoPars()),
        map_sets=[
            MapSetConfig(
                freq_tag=93,
                exp_tag="SO",
                beam=30.0,
                nhits_map_path="SO_nominal",
                obsmat_path="some/obsmat.npz",
            )
        ],
        map_sim_pars=MapSimConfig(filter_sims=True),
    )


def test_get_default_config(example_config: Config, tmp_path: Path) -> None:
    manager = DataManager(example_config)
    out_file = tmp_path / "default_config.yaml"
    manager.dump_config(out_file)

    assert out_file.exists()

    # Check that the file is not empty
    content = out_file.read_text()
    assert content.strip() != ""

    # Checking for the presence of a couple of key words, not sure if useful...
    content = out_file.read_text()
    assert "output_dirs" in content
    assert "data_dirs" in content
