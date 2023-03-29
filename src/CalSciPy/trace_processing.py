from __future__ import annotations
import numpy as np
from scipy.signal import firwin, filtfilt
from typing import Optional


def calculate_dfof(traces: np.ndarray, frame_rate: float = 30.0, in_place: bool = False,
                   offset: float = 0.0, external_reference: Optional[np.ndarray] = None) \
        -> np.ndarray:
    """
    Calculates Δf/f0 (fold fluorescence over baseline). Baseline is defined as the 5th percentile of the signal
    after a 1Hz low-pass filter using a Hamming window. Baseline can be calculated using an external reference using the
    raw argument or adjusted by using the offset argument. Supports in-place calculation (off by default).

    :param traces: matrix of traces in the form of neurons x frames
    :type traces: numpy.ndarray
    :param frame_rate: frame rate of dataset
    :type frame_rate: float = 30.0
    :param in_place: boolean indicating whether to perform calculation in-place
    :type in_place: bool = False
    :param offset: offset added to baseline; useful if traces are non-negative
    :type offset: float = 0.0
    :param external_reference: secondary dataset used to calculate baseline; useful if traces have been factorized
    :type external_reference: numpy.ndarray = None
    :return: Δf/f0 matrix of n neurons x m samples
    :rtype: numpy.ndarray
    """
    if in_place:
        dfof = traces
    else:
        dfof = traces.copy()

    taps = 30  # More taps mean higher frequency resolution, which in turn means narrower filters and/or steeper
    # roll‐offs.
    filter_frequency = 1  # (Hz)
    baseline_percentile = 5
    neurons, samples = dfof.shape

    # if for some reason sample is less than 90 frames we'll reduce the number of taps
    # we'll also make sure it's odd so we always have type I linear phase
    taps = min(taps, int(samples // 3))
    if taps % 2 == 0:
        taps -= 1

    # determine padding length, and if padding == samples then further reduce taps
    padding = 3 * taps
    if padding == samples:
        taps -= 1
        padding = 3 * taps

    # hamming window
    filter_window = firwin(taps, cutoff=filter_frequency, fs=frame_rate)

    for neuron in range(neurons):
        if external_reference is not None:
            # filter
            filtered_trace = filtfilt(filter_window, [1.0], external_reference[neuron, :], axis=0, padlen=padding)
            # calculate baseline
            baseline = np.percentile(filtered_trace, baseline_percentile, axis=0, keepdims=True) + offset
        else:
            # filter
            filtered_trace = filtfilt(filter_window, [1.0], dfof[neuron, :], axis=0, padlen=padding)
            # calculate baseline
            baseline = np.percentile(filtered_trace, baseline_percentile, axis=0, keepdims=True) + offset

        # calculate dfof
        dfof[neuron, :] = (dfof[neuron, :] - baseline) / baseline

    return dfof


def calculate_standardized_noise(fold_fluorescence_over_baseline: np.ndarray, frame_rate: float = 30.0) -> np.ndarray:
    """
    Calculates a frame-rate independent standardized noise as defined as:
        | :math:`v = \\frac{\sigma \\frac{\Delta F}F}\sqrt{f}`

    It is robust against outliers and approximates the standard deviation of Δf/f0 baseline fluctuations.
    For comparison, the more exquisite of the Allen Brain Institute's public datasets are approximately 1*%Hz^(-1/2)

    :param fold_fluorescence_over_baseline: fold fluorescence over baseline (i.e., Δf/f0)
    :type fold_fluorescence_over_baseline: numpy.ndarray
    :param frame_rate: frame rate of dataset
    :type frame_rate: float = 30
    :return: standardized noise (units are  1*%Hz^(-1/2) ) for each neuron
    :rtype: numpy.ndarray
    """
    return 100.0 * np.median(np.abs(np.diff(fold_fluorescence_over_baseline, axis=1)), axis=1) / np.sqrt(frame_rate)


def detrend_polynomial(traces: np.ndarray, in_place: bool = False) -> np.ndarray:
    """
    Detrend traces using a fourth-order polynomial

    :param traces: matrix of traces in the form of neurons x frames
    :type traces: numpy.ndarray
    :param in_place: boolean indicating whether to perform calculation in-place
    :type in_place: bool = False
    :return: detrended traces
    :rtype: numpy.ndarray
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
