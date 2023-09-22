from __future__ import annotations
from typing import Callable

import numpy as np


def baseline_calculation(method: str) -> Callable:
    if method == "mean":
        return mean_baseline


def mean_baseline(traces: np.ndarray) -> np.ndarray:
    """
    Calculates baseline as the mean of all observations for each traces

    :param traces: Matrix of n neurons x m samples

    :returns: Vector of n neurons where each element is the baseline for the associated neuron
    """
    return np.mean(traces, axis=-1) * np.ones(traces.shape)
