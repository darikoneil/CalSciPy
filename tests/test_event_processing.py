import pytest
import numpy as np
from CalSciPy.event_processing import bin_events, calculate_firing_rates, calculate_mean_firing_rates, \
    gaussian_smooth_firing_rates, normalize_firing_rates
from PPVD.style import TerminalStyle
from itertools import product

# samples
samples_event_matrix = np.full((5, 30), 1)
samples_event_matrix[0, :] = np.arange(30)
samples_event_matrix[1, np.arange(0, 30, 3)] = 3
samples_event_matrix[2, :] = np.arange(30)
samples_event_matrix[1, np.arange(0, 30, 2)] = 2
samples_event_matrix[4, :] = np.arange(30)
bin_length_ = [2, 3, 5, 10]
sample_binning = tuple(list(product([samples_event_matrix], bin_length_)))


@pytest.mark.parametrize(("matrix", "bin_lengths"), [*sample_binning])
def test_bin_events_passes(matrix, bin_lengths):
    binned_matrix = bin_events(matrix, bin_lengths)
    if binned_matrix.shape[0] != matrix.shape[0]:
        raise AssertionError(f"{TerminalStyle.YELLOW} Binned matrix lost features {TerminalStyle.RESET}")
    if not binned_matrix.shape[1] < matrix.shape[1]:
        raise AssertionError(f"{TerminalStyle.YELLOW} Binned matrix is the same shape as the original matrix "
                             f"{TerminalStyle.RESET}")


def test_calculate_firing_rates():
    pass


def test_mean_firing_rates():
    pass


def test_gaussian_smooth_firing_rates():
    pass


def test_normalize_firing_rates():
    pass
