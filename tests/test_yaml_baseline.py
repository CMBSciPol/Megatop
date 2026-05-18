"""Baseline YAML round-trip test.

Loads every paramfiles/*.yaml via Config.load_yaml, dumps via Config.dump_yaml,
re-parses both sides as plain dicts, and asserts dict-equality. Acts as the
oracle for the attrs/cattrs -> pydantic migration: any phase that breaks
existing paramfiles will surface here.
"""

from pathlib import Path

import pytest
import yaml

from megatop.config import Config

PARAMFILES_DIR = Path(__file__).parent.parent / "paramfiles"
PARAMFILES = sorted(PARAMFILES_DIR.glob("*.yaml"))


@pytest.mark.parametrize("paramfile", PARAMFILES, ids=[p.name for p in PARAMFILES])
def test_paramfile_roundtrips(paramfile: Path, tmp_path: Path) -> None:
    cfg1 = Config.load_yaml(paramfile)
    out1 = tmp_path / "first.yaml"
    cfg1.dump_yaml(out1)

    cfg2 = Config.load_yaml(out1)
    out2 = tmp_path / "second.yaml"
    cfg2.dump_yaml(out2)

    assert yaml.safe_load(out1.read_text()) == yaml.safe_load(out2.read_text())
