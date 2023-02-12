from __future__ import annotations
import numpy as np


def bin_data():
    return
# TODO PASTE, DOCUMENT, UNIT TEST


def gaussian_smooth_firing_rates(firing_matrix: np.ndarray) -> np.ndarray:
    return
# TODO PASTE, DOCUMENT, UNIT TEST


def calculate_firing_rate(spike_probability_matrix: np.ndarray, frame_rate: float = 30) -> np.ndarray:
    """

    :param spike_probability_matrix: matrix of n neuron x m samples where each element is the probability of a spike
    :type spike_probability_matrix: np.ndarray
    :param frame_rate: frame rate of dataset
    :type frame_rate: float = 30
    :return: matrix of firing rates
    :rtype: np.ndarray
    """
    return spike_probability_matrix * frame_rate
# TODO UNIT TEST


def calculate_mean_firing_rate(firing_matrix: np.ndarray) -> np.ndarray:
    """
    Calculate mean firing rate

    :param firing_matrix: matrix of n neuron x m samples where each element is either a spike or an
    instantaneous firing rate
    :type firing_matrix: np.ndarray
    :return: 1-D vector of mean firing rates
    :rtype: np.ndarray
    """
    return np.nanmean(NeuralData, axis=1)
# TODO UNIT TEST


def normalize_firing_rates(firing_matrix: np.ndarray) -> np.ndarray:
    return
# TODO PASTE, DOCUMENT, UNIT TEST

