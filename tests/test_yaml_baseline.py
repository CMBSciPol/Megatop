"""Baseline YAML round-trip test.

Loads every paramfiles/*.yaml via Config.load_yaml, dumps via Config.dump_yaml,
reloads, and asserts Config equality.
"""

from pathlib import Path

import pytest

from megatop.config import Config

PARAMFILES_DIR = Path(__file__).parent.parent / "paramfiles"
PARAMFILES = sorted(PARAMFILES_DIR.glob("*.yaml"))


@pytest.mark.parametrize("paramfile", PARAMFILES, ids=[p.name for p in PARAMFILES])
def test_paramfile_roundtrips(paramfile: Path, tmp_path: Path) -> None:
    cfg1 = Config.load_yaml(paramfile)
    out = tmp_path / "roundtrip.yaml"
    cfg1.dump_yaml(out)
    cfg2 = Config.load_yaml(out)
    assert cfg1 == cfg2
