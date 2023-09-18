from __future__ import annotations

import numpy as np


def calculate_standardized_noise(fold_fluorescence_over_baseline: np.ndarray, frame_rate: float = 30.0) -> np.ndarray:
    """
    Calculates a frame-rate independent standardized noise as defined as:
        | :math:`v = \\frac{\sigma \\frac{\Delta F}F}\sqrt{f}`

    It is robust against outliers and approximates the standard deviation of Δf/f0 baseline fluctuations.
    For comparison, the more exquisite of the Allen Brain Institute's public datasets are approximately 1*%Hz^(-1/2)

    :param fold_fluorescence_over_baseline: fold fluorescence over baseline (i.e., Δf/f0)

    :param frame_rate: frame rate of dataset

    :returns: standardized noise (1*%Hz^(-1/2) ) for each neuron
    """
    return 100.0 * np.median(np.abs(np.diff(fold_fluorescence_over_baseline, axis=1)), axis=1) / np.sqrt(frame_rate)


def detrend_polynomial(traces: np.ndarray, in_place: bool = False) -> np.ndarray:
    """
    Detrend traces using a fourth-order polynomial

    :param traces: matrix of traces in the form of neurons x frames

    :param in_place: boolean indicating whether to perform calculation in-place

    :returns: detrended traces
    """
    [_neurons, _samples] = traces.shape
    _samples_vector = np.arange(_samples)

    if in_place:
        detrended_matrix = traces
    else:
        detrended_matrix = traces.copy()

    for _neuron in range(_neurons):
        _fit = np.polyval(np.polyfit(_samples_vector, detrended_matrix[_neuron, :], deg=4), _samples_vector)
        detrended_matrix[_neuron] -= _fit

    return detrended_matrix
