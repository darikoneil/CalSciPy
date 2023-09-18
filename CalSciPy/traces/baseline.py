from __future__ import annotations
from typing import Optional

import numpy as np
from .._calculations import sliding_window
from scipy.signal import firwin, filtfilt
from scipy.ndimage.filters import gaussian_filter1d
from tqdm import tqdm


def calculate_dfof(traces: np.ndarray,
                   frame_rate: float = 30.0,
                   in_place: bool = False,
                   offset: float = 0.0,
                   external_reference: Optional[np.ndarray] = None,
                   method: str = "baseline"
                   ) -> np.ndarray:
    if method == "baseline":
        return _calculate_dfof_mean_of_percentile(traces, frame_rate, in_place, offset, external_reference)
    else:
        return _calculate_dfof_filter(traces, frame_rate, in_place, offset, external_reference)


def _calculate_dfof_filter(traces: np.ndarray, frame_rate: float = 30.0, in_place: bool = False,
                   offset: float = 0.0, external_reference: Optional[np.ndarray] = None) \
        -> np.ndarray:
    # noinspection GrazieInspection
    """
        Calculates Δf/f0 (fold fluorescence over baseline). Baseline is defined as the 5th percentile of the signal
        after a 1Hz low-pass filter using a Hamming window. Baseline can be calculated using an external reference
        | using the raw argument or adjusted by using the offset argument. Supports in-place calculation
        | (off by default).

        :param traces: matrix of traces in the form of neurons x frames
        :param frame_rate: frame rate of dataset
        :param in_place: boolean indicating whether to perform calculation in-place
        :param offset: offset added to baseline; useful if traces are non-negative
        :param external_reference: secondary dataset used to calculate baseline; useful if traces have been factorized
        :return: Δf/f0 matrix of n neurons x m samples
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


def _calculate_dfof_mean_of_percentile(traces: np.ndarray,
                                       frame_rate: float = 30.0,
                                       in_place: bool = False,
                                       offset: float = 0.0,
                                       external_reference: Optional[np.ndarray] = None
                                       ) -> np.ndarray:
    baseline = np.nanmean(sliding_window(traces, int(frame_rate * 30), np.nanpercentile, q=8, axis=-1), axis=0)
    if not in_place:
        dfof = np.zeros_like(traces)
    for neuron in range(dfof.shape[0]):
        dfof[neuron, :] = (traces[neuron, :] - baseline[neuron]) / baseline[neuron]
    return dfof
