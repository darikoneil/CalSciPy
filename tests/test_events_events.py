import pytest

import numpy as np

from CalSciPy.events import generate_raster


"""
Test for events (generic)
"""


@pytest.fixture(scope="function")
def spike_times(request):
    return (
        [0, 2, 4, 6, 8],
        [1, 3, 5, 7, 9],
        [3, 4, 5, 6, 7]
            )


@pytest.fixture(scope="function")
def expected_spike_times(request):
    expected = np.zeros((3, 10))
    expected[0, [0, 2, 4, 6, 8]] = 1
    expected[1, [1, 3, 5, 7, 9]] = 1
    expected[2, [3, 4, 5, 6, 7]] = 1
    return expected


def test_generate_raster(spike_times, expected_spike_times):
    # test standard
    spikes = generate_raster(spike_times, 10)
    np.testing.assert_equal(spikes, expected_spike_times)
    # test inferring num samples
    spikes_inferred = generate_raster(spike_times)
    np.testing.assert_equal(spikes_inferred, expected_spike_times)
