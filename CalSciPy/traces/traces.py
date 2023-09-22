from __future__ import annotations
from typing import Optional
from numbers import Number

import numpy as np

from .baseline import baseline_calculation


def calculate_dfof(traces: np.ndarray,
                   method: str = "mean",
                   external_reference: Optional[np.ndarray] = None,
                   offset: Number = 0,
                   in_place: bool = False,
                   **kwargs
                   ) -> np.ndarray:
    """
    Calculates the fold fluorescence over baseline (i.e., Δf/f0). A variety of different methods of calculating the
    baseline are provided. All keyword-arguments are passed to the baseline calculation.

    :param traces: Matrix of n neurons x m samples

    :param method: Method used to calculate baseline fluorescence

    :param external_reference: Matrix of n neurons x m samples used to calculate baseline

    :param offset: Used to offset baselines by some constant

    :param in_place: Whether to perform calculations in-place

    :returns: Matrix of n neurons x m samples where each element is the fold fluorescence over baseline for a
        particular observation.

    """

    # determine if in place
    if in_place:
        dfof = traces
    else:
        dfof = traces.copy()

    # point to external reference if not using original traces as the reference
    if external_reference is not None:
        reference_traces = external_reference
    else:
        reference_traces = traces

    # retrieve baseline calculation method
    baseline_func = baseline_calculation(method)

    # calculate baselines (Uses 2X memory, but much more simple?)
    baselines = baseline_func(reference_traces, **kwargs) + offset

    # calculate dfof
    dfof -= baselines
    dfof /= baselines

    return dfof


def calculate_standardized_noise(dfof: np.ndarray, frame_rate: Number = 30.0) -> np.ndarray:
    """
    Calculates a frame-rate independent standardized noise as defined as:
        | :math:`v = \\frac{\sigma \\frac{\Delta F}F}\sqrt{f}`

    It is robust against outliers and approximates the standard deviation of Δf/f0 baseline fluctuations.
    This metric was first defined in the publication associated with the spike-inference software package
    `CASCADE <https://www.nature.com/articles/s41593-021-00895-5>`_\.


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
