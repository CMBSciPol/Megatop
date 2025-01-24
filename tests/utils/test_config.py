import pytest

from megatop.config import _GeneralPars


def test_lmax_validator():
    with pytest.raises(ValueError, match="less than or equal to"):
        _GeneralPars(nside=128, lmax=500)
