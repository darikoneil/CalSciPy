import pytest
import numpy as np
from CalSciPy.event_processing import calculate_firing_rates, calculate_mean_firing_rates


# most of these are a bit trivial for tests but we'll make tests to check when/if refactoring

# samples bin events not great
samples_event_matrix = np.full((5, 30), 1)
samples_event_matrix[0, :] = np.arange(30)
samples_event_matrix[1, np.arange(0, 30, 3)] = 3
samples_event_matrix[2, :] = np.arange(30)
samples_event_matrix[1, np.arange(0, 30, 2)] = 2
samples_event_matrix[4, :] = np.arange(30)
bin_length_ = [2, 3, 5, 10]

# samples calculate firing rates
samples_frame_rate = [10, 10]
samples_spike_probabilities = np.array([
    [0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 2.20],
    [2.20, 0.75, 0.75, 0.5, 0.5, 0.25, 0.25],
    [0, 0, 0.1, 0.1, 0.75, 1, 1]
])
samples_spike_probabilities = [samples_spike_probabilities, samples_spike_probabilities]

# samples mean firing rates
samples_mean_firing_rate = np.array([
    [2, 2, 3, 4, 4],
    [1, 2, 3, 4, 5],
    [1, 1, 1.5, 2, 2]
])
samples_mean_firing_rate = [samples_mean_firing_rate, samples_mean_firing_rate]

# samples gaussian
samples_gauss = np.array([
    [0, 5, 10, 7.5, 10, 5, 0],
    [7.5, 8.9, 10.0, 3.0, 5.0, 7.0]
])
samples_gauss = [samples_gauss, samples_gauss]
samples_sigma = [1, 2]

# samples normalize
samples_norm = np.array([
    [0.0, 50, 100, 50, 0],
    [0, 25, 50, 75, 100],
    [0, 5, 10, 100, 2]
])
samples_norm = [samples_norm, samples_norm]


@pytest.mark.parametrize("spike_probabilities", samples_spike_probabilities)
@pytest.mark.parametrize("frame_rate", samples_frame_rate)
def test_calculate_firing_rates(spike_probabilities, frame_rate):
    firing_matrix = calculate_firing_rates(spike_probabilities, frame_rate)
    print(firing_matrix)
    print(spike_probabilities.shape)
    if firing_matrix.shape != spike_probabilities.shape:
        raise AssertionError("Firing matrix shape mutated")
    np.testing.assert_equal(firing_matrix, np.array([
        [2.5, 2.5, 5.0, 5.0, 7.5, 7.5, 22.0],
        [22.0, 7.5, 7.5, 5.0, 5.0, 2.5, 2.5],
        [0.0, 0.0, 1.0, 1.0, 7.5, 10.0, 10.0]
    ]), err_msg="Miscalculation of firing rate")


@pytest.mark.parametrize("matrix", samples_mean_firing_rate)
def test_mean_firing_rates(matrix):
    mean_firing_rates = calculate_mean_firing_rates(matrix)
    if mean_firing_rates.shape[0] != matrix.shape[0]:
        raise AssertionError("Mean firing rate matrix mutated")
    np.testing.assert_equal(mean_firing_rates, np.array([
        3.0, 3.0, 1.5
    ]), err_msg="Miscalculation of mean firing rates")
