import pytest
import numpy as np
from CalSciPy.event_processing import calculate_firing_rates, calculate_mean_firing_rates, \
    normalize_firing_rates


def test_calculate_firing_rates(spike_probabilities, frame_rate, expected_firing_rate):
    # out of place
    firing_matrix = calculate_firing_rates(spike_probabilities, frame_rate)
    np.testing.assert_equal(firing_matrix, expected_firing_rate)
    # in place
    calculate_firing_rates(spike_probabilities, frame_rate, in_place=True)
    np.testing.assert_equal(spike_probabilities, expected_firing_rate)


def test_calculate_mean_firing_rates(spike_probabilities, mean_firing_rates):
    mean_rates = calculate_mean_firing_rates(spike_probabilities)
    np.testing.assert_allclose(mean_rates, mean_firing_rates, rtol=1e-3)


def test_normalize_firing_rates(spike_probabilities, normalized_firing_rates):
    # out of place
    normal_rates = normalize_firing_rates(spike_probabilities)
    np.testing.assert_allclose(normal_rates, normalized_firing_rates, rtol=1e-3)
    # in place
    normalize_firing_rates(spike_probabilities, in_place=True)
    np.testing.assert_allclose(spike_probabilities, normalized_firing_rates, rtol=1e-3)