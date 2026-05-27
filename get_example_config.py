#!/usr/bin/env python3

from pathlib import Path

from megatop import Config, DataManager

default_config = Config.get_example()
manager = DataManager(default_config)
manager.dump_config(Path(__file__).parent / "paramfiles/default_config.yaml")
