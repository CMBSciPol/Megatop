from pathlib import Path

import megatop
from megatop import Config, DataManager

default_config = Config.get_example()

megatop_path = Path(megatop.__path__[0]).parents[1]
paramfiles_path = megatop_path / "paramfiles"
manager = DataManager(default_config)
manager.dump_config(paramfiles_path / "default_config.yaml")
