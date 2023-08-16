import pytest
import numpy as np
from CalSciPy.events import calculate_instantaneous_firing_rates, calculate_mean_firing_rates, \
    normalize_firing_rates


@pytest.fixture(scope="function")
def spike_probabilities(request):
    """
    This is a simple spike probability matrix of 3 neurons with 7 samples each,
    hardcoded to save space in the source repo.

    """
    return np.array([
        [0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 2.20],
        [2.20, 0.75, 0.75, 0.5, 0.5, 0.25, 0.25],
        [0, 0, 0.1, 0.1, 0.75, 1, 1]
    ])


@pytest.fixture(scope="function")
def expected_instantaneous_firing_rate(request):
    """
    This is an instantaneous firing rate matrix of 3 neurons with 7 samples each,
    calculated using the above fixture, hardcoded to save space in the source repo.

    """
    return np.array([
        [7.5, 7.5, 15.0, 15.0, 22.5, 22.5, 66.0],
        [66.0, 22.5, 22.5, 15.0, 15.0, 7.5, 7.5],
        [0.0, 0.0, 3.0, 3.0, 22.5, 30.0, 30.0]
    ])


@pytest.fixture(scope="function")
def expected_mean_firing_rates(request):
    """
    This is the expected mean firing rate of 3 neurons with 7 samples each,
    calculated using the above fixture, hardcoded to save space in the source repo.

    """
    return np.array([0.74285714, 0.74285714, 0.42142857])


@pytest.fixture(scope="function")
def expected_normalized_firing_rates(request):
    """
    This is a normalized firing rate matrix of 3 neurons with 7 samples each,
    calculated using the above fixture, hardcoded to save space in the source repo.

    """
    return np.array([
        [0.11364, 0.33333, 0.66667, 1.00000, 1.00000, 0.75000, 1.00000],
        [1.00000, 1.00000, 1.00000, 1.00000, 0.66667, 0.25000, 0.11364],
        [0.00000, 0.00000, 0.13333, 0.20000, 1.00000, 1.00000, 0.45455]
    ])


def test_calculate_instantaneous_firing_rates(spike_probabilities, expected_instantaneous_firing_rate):
    frame_rate = 30.0
    # out of place
    firing_matrix = calculate_instantaneous_firing_rates(spike_probabilities, frame_rate)
    np.testing.assert_equal(firing_matrix, expected_instantaneous_firing_rate)
    # in place
    calculate_instantaneous_firing_rates(spike_probabilities, frame_rate, in_place=True)
    np.testing.assert_equal(spike_probabilities, expected_instantaneous_firing_rate)


def test_calculate_mean_firing_rates(spike_probabilities, expected_mean_firing_rates):
    mean_rates = calculate_mean_firing_rates(spike_probabilities)
    np.testing.assert_allclose(mean_rates, expected_mean_firing_rates, rtol=1e-3)


def test_normalize_firing_rates(spike_probabilities, expected_normalized_firing_rates):
    # out of place
    normal_rates = normalize_firing_rates(spike_probabilities)
    np.testing.assert_allclose(normal_rates, expected_normalized_firing_rates, rtol=1e-3)
    # in place
    normalize_firing_rates(spike_probabilities, in_place=True)
    np.testing.assert_allclose(spike_probabilities, expected_normalized_firing_rates, rtol=1e-3)
