from __future__ import annotations
import numpy as np


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
