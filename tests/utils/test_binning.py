import numpy as np
import pytest

from megatop.utils.binning import create_binning


@pytest.mark.parametrize(
    "lmax, delta_ell, uniform_start, expected_low, expected_high",
    [
        (
            100,
            20,
            None,
            [2, 20, 40, 60, 80],
            [19, 39, 59, 79, 100],
        ),
        (
            103,
            20,
            None,
            [2, 20, 40, 60, 80, 100],
            [19, 39, 59, 79, 99, 103],
        ),
        (
            200,
            40,
            30,
            [2, 30, 70, 110, 150, 190],
            [29, 69, 109, 149, 189, 200],
        ),
        (
            200,
            40,
            40,  # uniform_start aligns with delta_ell grid: no spurious extra bin
            [2, 40, 80, 120, 160],
            [39, 79, 119, 159, 200],
        ),
    ],
)
def test_create_binning(lmax, delta_ell, uniform_start, expected_low, expected_high):
    low, high, center = create_binning(lmax, delta_ell, uniform_start)
    assert np.array_equal(low, expected_low)
    assert np.array_equal(high, expected_high)
    assert np.allclose(center, (np.array(expected_low) + np.array(expected_high)) / 2)
