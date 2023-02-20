from __future__ import annotations
import numpy as np
from numba import jit
from scipy.signal import firwin, filtfilt
from typing import Optional


def calculate_dfof(traces: np.ndarray, frame_rate: float = 30, in_place: bool = False,
                   offset: float = 0.0, raw: Optional[np.ndarray] = None) \
        -> np.ndarray:
    """
    Calculates Δf/f0 (fold fluorescence over baseline). Baseline is defined as the 5th percentile of the signal
    after a 1Hz low-pass filter using a Hamming window.

    :param traces: matrix of traces in the form of neurons x frames
    :type traces: numpy.ndarray
    :param frame_rate: frame rate of dataset
    :type frame_rate: float = 30
    :param in_place: boolean indicating whether to perform calculation in-place
    :type in_place: bool = False
    :param offset: offset added to baseline; useful if traces are non-negative
    :type offset: float
    :param raw: raw dataset used to calculate baseline; useful if traces have been factorized
    :type raw: numpy.ndarray or None
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
    taps = min(taps, int(max(taps, samples/3)))
    if taps % 2 == 0:
        taps -= 1

    # determine padding length
    padding = 3 * taps

    # hamming window
    filter_window = firwin(taps, cutoff=filter_frequency, fs=frame_rate)

    for neuron in range(neurons):
        # filter
        filtered_trace = filtfilt(filter_window, [1.0], dfof[neuron, :], axis=0, padlen=padding)
        # calculate baseline
        baseline = np.percentile(filtered_trace, baseline_percentile, axis=0, keepdims=True)
        # calculate dfof
        dfof[neuron, :] = (dfof[neuron, :] - baseline) / baseline
    # TODO probably can do without loop, check scipy

    return dfof
# TODO DOCUMENT, UNIT TEST, IMPLEMENT RAW


@jit
def calculate_standardized_noise(fold_fluorescence_over_baseline: np.ndarray, frame_rate: float = 30) -> np.ndarray:
    """
    Calculates a frame-rate independent standardized noise as defined as:
        | :math:`v = \\frac{\sigma \\frac{\Delta F}F}\sqrt{f}`

    It is robust against outliers and approximates the standard deviation of Δf/f0 baseline fluctuations.
    For comparison, the more exquisite of the Allen Brain Institute's public datasets are approximately 1*%Hz^(-1/2)

    :param fold_fluorescence_over_baseline: fold fluorescence over baseline (i.e., Δf/f0)
    :type fold_fluorescence_over_baseline: numpy.ndarray
    :param frame_rate: frame rate of dataset
    :type frame_rate: float = 30
    :return: standardized noise (units are  1*%Hz^(-1/2) )
    :rtype: numpy.ndarray
    """
    return np.median(np.abs(np.diff(fold_fluorescence_over_baseline))) / np.sqrt(frame_rate)
# TODO UNIT TEST


@jit
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
# TODO UNIT TEST


@jit
def _perona_malik_diffusion(trace: np.ndarray, iters: int = 5, kappa: int = 100, gamma: float = 0.15) -> np.ndarray:
    """
    Edge-preserving smoothing using perona malik diffusion

    :param traces: a matrix of neurons x frames
    :type traces: numpy.ndarray
    :param iters: number of iterations
    :type iters: int = 5
    :param kappa: diffusivity conductance
    :type kappa: int = 100
    :param gamma: step size (must be less than 1)
    :type gamma: float = 0.15
    :return: smoothed traces
    :rtype: numpy.ndarray
    """
    return
# TODO PASTE ME, DOCUMENT, UNIT TEST


def smooth_perona_malik(traces: np.ndarray, iters: int = 5, kappa: int = 100, gamma: float = 0.15) -> np.ndarray:
    """
    Edge-preserving smoothing using perona malik diffusion

    :param traces: a matrix of neurons x frames
    :type traces: numpy.ndarray
    :param iters: number of iterations
    :type iters: int = 5
    :param kappa: diffusivity conductance
    :type kappa: int = 100
    :param gamma: step size (must be less than 1)
    :type gamma: float = 0.15
    :return: smoothed traces
    :rtype: numpy.ndarray
    """
    return
# TODO PASTE ME, DOCUMENT, UNIT TEST
