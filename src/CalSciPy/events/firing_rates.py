from __future__ import annotations
from numbers import Number
import numpy as np


def calc_firing_rates(spike_prob: np.ndarray,
                      frame_rate: Number = None,
                      bin_duration: Number = None,
                      in_place: bool = False
                      ) -> np.ndarray:
    """
    Calculate instantaneous firing rates from spike (event) probabilities.

    :param spike_prob: Matrix where each element is the probability of a spiking-event having occurred
        for that particular combination of neuron and sample.

    :param frame_rate: Frame rate of the dataset

    :param bin_duration: Duration of each observation in seconds

    :param in_place: Whether to perform calculation in-place

    :returns: Firing matrix of n neurons x m samples where each element indicating the instantaneous firing (event) rate

    .. note::

        The elements of a spike probability matrix ought to be non-negative, but they do not need to be constrained to
        values of less than or equal to one.

    .. note::

        A value can be provided for only one of *frame_rate* and *bin_duration*

    """

    # determine what to multiply by
    if frame_rate is not None and bin_duration is not None:
        raise AssertionError("Provide only one of frame_rate and bin_duration!")
    elif bin_duration is not None:
        multiplier = 1 / bin_duration
    else:
        multiplier = frame_rate

    if in_place:
        firing_matrix = spike_prob
    else:
        firing_matrix = spike_prob.copy()

    firing_matrix *= multiplier

    return firing_matrix


def calc_mean_firing_rates(firing_rates: np.ndarray) -> np.ndarray:
    """
    Calculate mean firing rates instantaneous firing rates.

    :param firing_rates: Matrix of n neuron x m samples or tensor of t trials x n neurons x m samples where each
        element is either a spike or an instantaneous firing rate

    :returns: 1-D vector of mean firing rates or 2-D matrix of trial-organized mean firing rates

    .. warning::

        If you do not pass firing_rates in the shape of n neurons x m samples or tensor of t trials x n neurons x
        m samples your result will be **wrong**.

    """
    return np.nanmean(firing_rates, axis=-1)


def normalize_firing_rates(firing_rates: np.ndarray,
                           in_place: bool = False
                           ) -> np.ndarray:
    """
    Normalize firing rates by scaling to a max of 1.0. Non-negativity constrained.

    :param firing_rates: Matrix of n neuron x m samples or tensor of t trials x n neurons x m samples where each
        element is either a spike or an instantaneous firing rate

    :param in_place: boolean indicating whether to perform calculation in-place

    :returns: normalized firing rate matrix of n neurons x m samples

    .. warning::

        If you do not pass firing_rates in the shape of n neurons x m samples or tensor of t trials x n neurons x
        m samples your result will be **wrong**.

    """

    if len(firing_rates) >= 3:
        raise NotImplementedError

    if in_place:
        normalized_matrix = firing_rates
    else:
        normalized_matrix = firing_rates.copy()

    # in place, transpose & divide
    np.divide(normalized_matrix.T, np.nanmax(normalized_matrix, axis=-1), normalized_matrix.T)

    return normalized_matrix
