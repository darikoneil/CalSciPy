from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from itertools import product


def bin_events(matrix: np.ndarray, bin_length: int) -> np.ndarray:
    """
    Bin events (e.g., spikes) using specified bin length

    :param matrix: matrix of n features x m samples
    :type matrix: numpy.ndarray
    :param bin_length: length of bin
    :type bin_length: int
    :return: binned_matrix of n features x m bins
    :rtype: numpy.ndarray
    """
    _features, _frames = matrix.shape
    _bins = pd.interval_range(0, _frames, freq=bin_length)
    binned_matrix = np.empty((_features, _frames // bin_length), dtype=np.float64)
    for _feature, _bin in product(range(_features), range(len(_bins))):
        binned_matrix[_feature, _bin] = \
            np.sum(matrix[_feature, int(_bins.values[_bin].left):int(_bins.values[_bin].right)])
    return binned_matrix


def calculate_firing_rates(spike_probability_matrix: np.ndarray, frame_rate: float = 30, in_place: bool = False) \
        -> np.ndarray:
    """
    Calculate firing rates

    :param spike_probability_matrix: matrix of n neuron x m samples where each element is the probability of a spike
    :type spike_probability_matrix: numpy.ndarray
    :param frame_rate: frame rate of dataset
    :type frame_rate: float = 30
    :param in_place: boolean indicating whether to perform calculation in-place
    :type in_place: bool = False
    :return: firing matrix of n neurons x m samples where each element is a binary indicating presence of spike event
    :rtype: numpy.ndarray
    """
    if in_place:
        firing_matrix = spike_probability_matrix
    else:
        firing_matrix = spike_probability_matrix.copy()

    firing_matrix *= frame_rate

    return firing_matrix
# TODO UNIT TEST


def calculate_mean_firing_rates(firing_matrix: np.ndarray) -> np.ndarray:
    """
    Calculate mean firing rate

    :param firing_matrix: matrix of n neuron x m samples where each element is either a spike or an
        instantaneous firing rate
    :type firing_matrix: numpy.ndarray
    :return: 1-D vector of mean firing rates
    :rtype: numpy.ndarray
    """
    return np.nanmean(firing_matrix, axis=1)
# TODO UNIT TEST


def gaussian_smooth_firing_rates(firing_matrix: np.ndarray, sigma: float, in_place: bool = False) -> np.ndarray:
    """
    Normalize firing rates using a 1-D gaussian filter

    :param firing_matrix: matrix of n neuron x m samples where each element is either a spike or an
        instantaneous firing rate
    :type firing_matrix: numpy.ndarray
    :param sigma: standard deviation of gaussian kernel
    :type sigma: float
    :param in_place: boolean indicating whether to perform calculation in-place
    :type in_place: bool = False
    :return: gaussian-smoothed firing rate matrix of n neurons x m samples
    :rtype: numpy.ndarray
    """
    if not in_place:
        return gaussian_filter1d(firing_matrix, sigma, axis=1)

    return gaussian_filter1d(firing_matrix, sigma, axis=1, output=firing_matrix)
# TODO UNIT TEST


def normalize_firing_rates(firing_matrix: np.ndarray, in_place: bool = False) -> np.ndarray:
    """
    Normalize firing rates by scaling to a max of 1.0. Non-negativity constrained.

    :param firing_matrix: matrix of n neuron x m samples where each element is either a spike or an
        instantaneous firing rate
    :type firing_matrix: numpy.ndarray
    :param in_place: boolean indicating whether to perform calculation in-place
    :type in_place: bool = False
    :return: normalized firing rate matrix of n neurons x m samples
    :rtype: numpy.ndarray
    """
    if in_place:
        normalized_matrix = firing_matrix
    else:
        normalized_matrix = firing_matrix.copy()

    normalized_matrix /= np.max(normalized_matrix, axis=0)
    normalized_matrix[normalized_matrix <= 0] = 0
    return normalized_matrix
# TODO UNIT TEST
