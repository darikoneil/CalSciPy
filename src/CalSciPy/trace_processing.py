from __future__ import annotations
import sys
import numpy as np
import itertools
from numba import jit


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


@jit
def calculate_standardized_noise(fold_fluorescence_over_baseline: np.ndarray, frame_rate: float = 30) -> np.ndarray:
    """
    Calculates a frame-rate independent standardized noise as defined as:
        | :math:`v = \\frac{\sigma \\frac{\Delta F}F}\sqrt{f}`

    It is robust against outliers and approximates the standard deviation of Δf/f0 baseline fluctuations.
    For comparison, the more exquisite of the Allen Brain Institute's public datasets are approximately 1*%Hz^(-1/2)

    :param fold_fluorescence_over_baseline: fold fluorescence over baseline (i.e., Δf/f0)
    :type fold_fluorescence_over_baseline: np.ndarray
    :param frame_rate: frame rate of dataset
    :type frame_rate: float = 30
    :return: standardized noise (units are  1*%Hz^(-1/2) )
    :rtype: np.ndarray
    """
    return np.median(np.abs(np.diff(fold_fluorescence_over_baseline))) / np.sqrt(frame_rate)
# TODO UNIT TEST


# TODO ADD PERONA MALIK DIFFUSION, NORMALIZED GAUSSIAN SMOOTHING, POLYNOMIAL DETRENDING, BASELINE CALCULATION
