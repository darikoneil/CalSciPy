from __future__ import annotations
from numbers import Number

import numpy as np


"""
Where all the helpers for visuals live
"""


def generate_time_vector(num_samples: int,
                         sampling_frequency: Number = 30.0,
                         start: Number = 0.0,
                         step: Number = None
                         ) -> np.ndarray:
    """
    Generates a time vector for a number of samples collected at either

    """

    if not step:
        step = 1 / sampling_frequency

    return np.arange(0, num_samples) * step + start
