from __future__ import annotations
from typing import Union, Iterable, Callable
from collections import namedtuple
from functools import wraps
import sys


import numpy as np

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt   # noqa: E402
from matplotlib.widgets import Slider  # noqa: E402
import seaborn as sns  # noqa: F401, E402


class _Colors:
    blue: Tuple[float, float, float] = (15/255, 159/255, 255/255)
    orange: Tuple[float, float, float] = (255/255, 159/255, 15/255)
    green: Tuple[float, float, float] = (64/255, 204/255, 139/255)
    red: Tuple[float, float, float] = (255/255, 78/255, 75/255)
    purple: Tuple[float, float, float] = (120/255, 64/255, 204/255)
    yellow: Tuple[float, float, float] = (255/255, 240/255, 15/255)
    light: Tuple[float, float, float] = (128/255, 128/255, 128/255)
    dark: Tuple[float, float, float] = (65/255, 65/255, 65/255)
    black: Tuple[float, float, float] = (0/255, 0/255, 0/255)
    white: Tuple[float, float, float] = (255/255, 255/255, 255/255)
    background: Tuple[float, float, float] = (245/255, 245/255, 245/255)
    mapping = list(enumerate([red, green, blue, orange, purple, yellow, light, dark, black, white, background]))

    def __call__(self, value: Union[int, str], *args, **kwargs):
        if isinstance(value, int):
            if value >= len(self.mapping):
                value = len(self.mapping) % value
            return self.mapping[value][1]
        elif isinstance(value, str):
            return getattr(self, value)

# instance
COLORS = _Colors()


def _set_context(function: Callable, *args, **kwargs) -> Callable:
    def decorator(*args, **kwargs):
        with plt.style.context("seaborn-darkgrid"):
            return function(*args, **kwargs)
    return decorator


@_set_context
def interactive_traces(traces: Union[np.ndarray, Iterable[np.ndarray]],
                       frame_rate: float = None,
                       y_units: str = "Δf/f0",
                       mode: str = "overlay") -> None:
    """
    Function to interactive compare traces. Press Up/Down to switch neurons

    :param traces: primary traces
    :param frame_rate: frame rate
    :param mode: mode to plot group
    :returns: interactive figure
    """
    if isinstance(traces, np.ndarray):
        datasets = 1
        neurons, frames = traces.shape
    else:
        datasets = len(traces)
        neurons, frames = traces[0].shape

    title_template = f"Trace: Neuron "

    if frame_rate:
        x = np.arange(0, frames / frame_rate, 1 / frame_rate)
        x_units = "Time (s)"
    else:
        x = np.arange(0, frames, 1)
        x_units = "Frame (#)"

    current_neuron = 0

    f = plt.figure(figsize=(16, 9))

    if datasets > 1:
        pass
    else:
        a = f.add_subplot(111)
        a.set_title(title_template + f"{current_neuron}")
        a.set_xlabel(x_units)
        a.set_ylabel(y_units)
        a.plot(x, traces[current_neuron, :])
