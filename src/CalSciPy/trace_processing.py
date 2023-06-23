from __future__ import annotations
from typing import Optional

import numpy as np
from scipy.signal import firwin, filtfilt
from scipy.ndimage.filters import gaussian_filter1d
from numba import njit
from tqdm import tqdm
from slide import sliding_window


def calculate_dfof(traces: np.ndarray, frame_rate: float = 30.0, in_place: bool = False,
                   offset: float = 0.0, external_reference: Optional[np.ndarray] = None, method="baseline") -> np.ndarray:
    if method == "baseline":
        return _calculate_dfof_baseline(traces, frame_rate, in_place, offset, external_reference)
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


def _calculate_dfof_baseline(traces: np.ndarray, frame_rate: float = 30.0, in_place: bool = False,
                   offset: float = 0.0, external_reference: Optional[np.ndarray] = None):
    baselines = np.nanmean(sliding_window(traces, frame_rate * 30, np.nanpercentile, 8, axis=-1), axis=0)
    if not in_place:
        dfof = np.zeros_like(traces)
    for neuron in range(dfof.shape[0]):
        dfof[neuron, :] = (traces[neuron, :] - baseline[neuron]) / baseline[neuron]
    return dfof


def calculate_standardized_noise(fold_fluorescence_over_baseline: np.ndarray, frame_rate: float = 30.0) -> np.ndarray:
    """
    Calculates a frame-rate independent standardized noise as defined as:
        | :math:`v = \\frac{\sigma \\frac{\Delta F}F}\sqrt{f}`

    It is robust against outliers and approximates the standard deviation of Δf/f0 baseline fluctuations.
    For comparison, the more exquisite of the Allen Brain Institute's public datasets are approximately 1*%Hz^(-1/2)

    :param fold_fluorescence_over_baseline: fold fluorescence over baseline (i.e., Δf/f0)
    :param frame_rate: frame rate of dataset
    :returns: standardized noise (units are  1*%Hz^(-1/2) ) for each neuron
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


def perona_malik_diffusion(traces: np.ndarray, iters: int = 25, kappa: float = 0.15, gamma: float = 0.25, sigma: float = 0,
                           in_place: bool = False) -> np.ndarray:
    """
    Edge-preserving smoothing using perona malik diffusion. This is a non-linear smoothing technique that avoids the
    temporal distortion introduced onto traces by standard gaussian smoothing.

    The parameter `kappa` controls the level of smoothing ("diffusion") as a function of the derivative of the trace
    (or "gradient" in the case of 2D images where this algorithm is often used). This function is known as the
    diffusion coefficient. When the derivative for some portion of the trace is low, the algorithm will encourage
    smoothing to reduce noise. If the derivative is large like during a burst of activity, the algorithm will discourage
    smoothing to maintain its structure. Here, the argument `kappa` is multiplied by the dynamic range to generate the
    true kappa.

    The diffusion coefficient implemented here is e^(-(derivative/kappa)^2).

    Perona-Malik diffusion is an iterative process. The parameter `gamma` controls the rate of diffusion, while
    parameter `iters` sets the number of iterations to perform.

    This implementation is currently situated to handle 1-D vectors because it gives us some performance benefits.

    :param traces: matrix of M neurons by N samples
    :param iters: number of iterations
    :param kappa: used to calculate the true kappa, where true kappa = kappa * dynamic range. range 0-1
    :param gamma: rate of diffusion for each iter. range 0-1
    :param in_place: whether to calculate in-place
    :returns: smoothed traces
    """

    assert (0 < iters), "iters must be at least 1"
    assert (0 <= kappa <= 1), "kappa must satisfy 0 <= kappa <= 1"
    assert (0 <= gamma <= 1), "gamma must satisfy 0 <= gamma <= 1"

    if in_place:
        smoothed_traces = traces
    else:
        smoothed_traces = traces.copy()

    if traces.ndim == 1:
        return _perona_malik_diffusion(traces, iters, kappa, gamma, sigma, in_place)
    else:
        for neuron in tqdm(range(traces.shape[0]), total=traces.shape[0], desc="Smoothing Traces"):
            smoothed_traces[neuron, :] = _perona_malik_diffusion(traces[neuron, :], iters, kappa, gamma, sigma, in_place)

    return smoothed_traces


def _perona_malik_diffusion(trace: np.ndarray, iters: int = 25, kappa: float = 0.15, gamma: float = 0.25, sigma: float = 0,
                            in_place: bool = False) -> np.ndarray:
    """
    Edge-preserving smoothing using perona malik diffusion. This is a non-linear extension of gaussian smoothing
    technique that avoids the temporal distortion introduced onto traces by standard gaussian smoothing.

    The parameter `kappa` controls the level of smoothing ("diffusion") as a function of the derivative of the trace
    (or "gradient" in the case of 2D images where this algorithm is often used). This function is known as the
    diffusion coefficient. When the derivative for some portion of the trace is low, the algorithm will encourage
    smoothing to reduce noise. If the derivative is large like during a burst of activity, the algorithm will discourage
    smoothing to maintain its structure. Here, the argument `kappa` is multiplied by the dynamic range to generate the
    true kappa.

    The diffusion coefficient implemented here is e^(-(derivative/kappa)^2). If the diffusion coefficient was a
    scalar instead of a function of the derivative the algorithm would be equivalent to standard gaussian smoothing.

    Perona-Malik diffusion is an iterative process. The parameter `gamma` controls the rate of diffusion, while
    parameter `iters` sets the number of iterations to perform.

    This implementation is currently situated to handle 1-D vectors because it gives us some performance benefits.

    :param trace: trace for a single neuron (i.e., vector)
    :param iters: number of iterations
    :param kappa: used to calculate the true kappa, where true kappa = kappa * dynamic range. range 0-1
    :param gamma: rate of diffusion for each iter. range 0-1
    :param in_place: whether to calculate in-place
    :returns: smoothed traces
    """

    if in_place:
        smoothed_trace = trace
    else:
        smoothed_trace = trace.copy()

    kappa = kappa * (np.max(smoothed_trace) - np.min(smoothed_trace))

    # preallocate
    derivative = np.zeros_like(smoothed_trace)

    for _ in range(iters):
        derivative = np.zeros_like(smoothed_trace)
        derivative[:-1] = np.diff(smoothed_trace)
        if sigma > 0:
            derivative = gaussian_filter1d(derivative, sigma)
        diffusion = derivative * _diffusion_coefficient(derivative, kappa)
        diffusion[1:] -= diffusion[:-1]
        smoothed_trace += (gamma * diffusion)

    return smoothed_trace


@njit
def _diffusion_coefficient(derivative: np.ndarray, kappa: float) -> np.ndarray:
    """
    Diffusion coefficient

    :param derivative: derivative
    :param kappa: kappa
    :returns: value
    """
    return np.exp(-((derivative / kappa)**2.0))
