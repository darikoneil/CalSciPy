from __future__ import annotations
import numpy as np
from scipy.ndimage import gaussian_filter1d


def bin_data():
    return
# TODO PASTE, DOCUMENT, UNIT TEST


def calculate_firing_rates(spike_probability_matrix: np.ndarray, frame_rate: float = 30, in_place: bool = False) \
        -> np.ndarray:
    """

    :param spike_probability_matrix: matrix of n neuron x m samples where each element is the probability of a spike
    :type spike_probability_matrix: np.ndarray
    :param frame_rate: frame rate of dataset
    :type frame_rate: float = 30
    :param in_place: boolean indicating whether to perform calculation in-place
    :type in_place: bool = False
    :return: matrix of firing rates
    :rtype: np.ndarray
    """
    if not in_place:
        firing_matrix = spike_probability_matrix.copy() * frame_rate
        return firing_matrix

    spike_probability_matrix *= frame_rate
    return spike_probability_matrix
# TODO UNIT TEST


def calculate_mean_firing_rates(firing_matrix: np.ndarray) -> np.ndarray:
    """
    Calculate mean firing rate

    :param firing_matrix: matrix of n neuron x m samples where each element is either a spike or an
    instantaneous firing rate
    :type firing_matrix: np.ndarray
    :return: 1-D vector of mean firing rates
    :rtype: np.ndarray
    """
    return np.nanmean(firing_matrix, axis=1)
# TODO UNIT TEST


def gaussian_smooth_firing_rates(firing_matrix: np.ndarray, sigma: float, in_place: bool = False) -> np.ndarray:
    """
    Normalize firing rates using a 1-D gaussian filter (simple calls :ref:`scipy.ndimage.gaussian_filter1d`

    :param firing_matrix: matrix of n neuron x m samples where each element is either a spike or an
    instantaneous firing rate
    :type firing_matrix: np.ndarray
    :param sigma: standard deviation of gaussian kernel
    :type sigma: float
    :param in_place: boolean indicating whether to perform calculation in-place
    :type in_place: bool = False
    :return: gaussian-smoothed firing rate matrix of n neurons x m samples
    :rtype: np.ndarray
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
    :type firing_matrix: np.ndarray
    :param in_place: boolean indicating whether to perform calculation in-place
    :type in_place: bool = False
    :return: normalized firing rate matrix of n neurons x m samples
    :rtype: np.ndarray
    """
    if not in_place:
        normalized_matrix = firing_matrix / np.max(firing_matrix, axis=0)
        normalized_matrix[normalized_matrix <= 0] = 0
        return normalized_matrix

    firing_matrix /= np.max(firing_matrix, axis=0)
    firing_matrix[firing_matrix <= 0] = 0
    return firing_matrix
# TODO UNIT TEST
