import pytest

from megatop.utils.config import _GeneralPars


def test_lmax_validator():
    with pytest.raises(ValueError, match="should be lower or equal to"):
        _GeneralPars(nside=128, lmax=500)
