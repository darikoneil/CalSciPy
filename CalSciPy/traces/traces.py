from __future__ import annotations
from numbers import Number

import numpy as np


def calculate_standardized_noise(dfof: np.ndarray, frame_rate: Number = 30.0) -> np.ndarray:
    """
    Calculates a frame-rate independent standardized noise as defined as:
        | :math:`v = \\frac{\sigma \\frac{\Delta F}F}\sqrt{f}`

    It is robust against outliers and approximates the standard deviation of Δf/f0 baseline fluctuations.
    This metric was first defined in the publication associated with the spike-inference software package
    `CASCADE <https://github.com/HelmchenLabSoftware/Cascade>`_\.


    :param dfof: Fold fluorescence over baseline (i.e., Δf/f0)

    :param frame_rate: Frame rate at which the images were acquired.

    :returns: Standardized noise (1*%Hz^(-1/2) ) for each neuron

    .. note::

        For comparison, the more exquisite of the Allen Brain Institute's public datasets are approximately 1*%Hz^(-1/2)

    .. warning::

        If you do not pass traces in the shape of n neurons x m samples your result will be incorrect.

    """
    return 100.0 * np.median(np.abs(np.diff(dfof, axis=1)), axis=1) / np.sqrt(frame_rate)


def detrend_polynomial(traces: np.ndarray, in_place: bool = False) -> np.ndarray:
    """
    Detrend traces using a fourth-order polynomial. This function is useful to correct for a drifting baseline due to
    photo-bleaching and other processes that cause time-dependent degradation of signal-to-noise.

    :param traces: Matrix of n neuron x m samples

    :param in_place: Whether to perform calculation in-place

    :returns: Detrended matrix of n neuron x m samples.

    .. warning::

        If you do not pass traces in the shape of n neurons x m samples your result will be incorrect.

    """
    neurons, samples = traces.shape
    samples_vector = np.arange(samples)

    if in_place:
        detrended_matrix = traces
    else:
        detrended_matrix = traces.copy()

    for neuron in range(neurons):
        fit = np.polyval(np.polyfit(samples_vector, detrended_matrix[neuron, :], deg=4), samples_vector)
        detrended_matrix[neuron] -= fit

    return detrended_matrix
