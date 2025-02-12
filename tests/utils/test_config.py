import pytest

from megatop.config import GeneralConfig


def test_lmax_validator():
    with pytest.raises(ValueError, match="less than or equal to"):
        GeneralConfig(nside=128, lmax=500)
