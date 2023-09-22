from __future__ import annotations
from typing import Callable
from numbers import Number

import numpy as np
from scipy.signal import firwin, filtfilt


def baseline_calculation(method: str) -> Callable:
    """
    Retrieves appropriate baseline calculation function

    :param method: Desired baseline calculation method

    :returns: The function for desired baseline calculation method
    """
    if method == "low-pass":
        return low_pass_baseline
    elif method == "mean":
        return mean_baseline


def low_pass_baseline(traces: np.ndarray,
                      frame_rate: Number = 30.0,
                      frequency: Number = 1.0,
                      percentile: Number = 5.0,
                      ) -> np.ndarray:

    """
    Calculates baseline as the x-th percentile of fluorescence after low-pass filtering using a Hamming window.

    :param traces: Matrix of n neurons x m samples

    :param frame_rate: Frame rate at which the images were acquired.

    :param frequency: Filter frequency (Hz)

    :param percentile: Percentile considered baseline

    :returns: Matrix of n neurons x m samples where each element is the sample's baseline value for the associated
        neuron
    """
    taps = 30  # More taps mean higher frequency resolution, which in turn means narrower filters and/or steeper
    # roll‐offs.

    neurons, samples = traces.shape

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
    filter_window = firwin(taps, cutoff=frequency, fs=frame_rate)

    # pre-allocate
    baselines = np.zeros_like(traces)

    # calculate baselines
    for neuron in range(neurons):
        filtered_trace = filtfilt(filter_window, [1.0], traces[neuron, :], axis=0, padlen=padding)
        baselines[neuron, :] = np.percentile(filtered_trace, percentile, axis=0, keepdims=True)

    return baselines


def mean_baseline(traces: np.ndarray) -> np.ndarray:
    """
    Calculates baseline as the mean of all observations for each traces

    :param traces: Matrix of n neurons x m samples

    :returns: Matrix of n neurons x m samples where each element is the baseline for the associated neuron
    """
    # must reshape to be 2D here to match shape of traces
    return np.mean(traces, axis=-1).reshape((traces.shape[0], 1)) * np.ones(traces.shape)
