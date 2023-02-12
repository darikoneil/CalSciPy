from __future__ import annotations
import numpy as np
from numba import jit
from obspy.signal.detrend import polynomial


def calculate_dfof(traces: np.ndarray) -> np.ndarray:
    return
# TODO PASTE ME, DOCUMENT, UNIT TEST


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


@jit
def detrend_polynomial(traces: np.ndarray) -> np.ndarray:
    """
    Detrend traces using a fourth-order polynomial

    :param traces: matrix of traces in the form of neurons x frames
    :type traces: np.ndarray
    :return: detrended traces
    :rtype: np.ndarray
    """
    [_neurons, _samples] = traces.shape
    _samples_vector = np.arange(_samples)

    for _neuron in range(_neurons):
        _fit = np.polyval(np.polyfit(_samples_vector, traces[_neuron, :], deg=4), _samples_vector)
        traces[_neuron] -= fit  # TODO AM I IN-PLACE? NEVER MATTERED BUT COULD. TEST ME

    return traces
# TODO UNIT TEST


@jit
def _perona_malik_diffusion(trace: np.ndarray, iters: int = 5, kappa: int = 100, gamma: float = 0.15) -> np.ndarray:
    """
    Edge-preserving smoothing using perona malik diffusion

    :param traces: a matrix of neurons x frames
    :type traces: np.ndarray
    :param iters: number of iterations
    :type iters: int = 5
    :param kappa: diffusivity conductance
    :type kappa: int = 100
    :param gamma: step size (must be less than 1)
    :type gamma: float = 0.15
    :return: smoothed traces
    :rtype: np.ndarray
    """
    return
# TODO PASTE ME, DOCUMENT, UNIT TEST


def smooth_perona_malik(traces: np.ndarray, iters: int = 5, kappa: int = 100, gamma: float = 0.15) -> np.ndarray:
    """
    Edge-preserving smoothing using perona malik diffusion

    :param traces: a matrix of neurons x frames
    :type traces: np.ndarray
    :param iters: number of iterations
    :type iters: int = 5
    :param kappa: diffusivity conductance
    :type kappa: int = 100
    :param gamma: step size (must be less than 1)
    :type gamma: float = 0.15
    :return: smoothed traces
    :rtype: np.ndarray
    """
    return
# TODO PASTE ME, DOCUMENT, UNIT TEST
