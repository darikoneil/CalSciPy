from __future__ import annotations
from typing import Union, Iterable, Callable
from collections import namedtuple
from functools import wraps
import sys
from abc import ABC, abstractmethod

import numpy as np


import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib import pyplot as plt
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

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(_Colors, cls).__new__(cls)
        return cls.instance

    def __call__(self, value: Union[int, str], *args, **kwargs):
        if isinstance(value, int):
            if value >= len(self.mapping):
                value = len(self.mapping) % value
            return self.mapping[value][1]
        elif isinstance(value, str):
            return getattr(self, value)


# instance
COLORS = _Colors()


class InteractivePlot(ABC):

    required_attrs = ["x_label", "y_label", "title_template"]

    def __init__(self):
        self.figure = None
        self.axes = None
        self.pointer = None
        self.__post_init__()

    def __post_init__(self):
        required_attributes = self.required_attrs
        for attr in required_attributes:
            if not hasattr(self, attr):
                raise AttributeError(f"Missing attribute: {attr}")

        self.init_figure()

    @property
    def title(self):
        return self.title_template + f"{self.pointer}"

    @abstractmethod
    def loop(self, event):
        ...

    def on_key(self, event):
        sys.stdout.flush()
        self.loop(event)
        self.figure.canvas.draw()

    @abstractmethod
    def set_limits(self):
        ...

    def init_figure(self):
        self.pointer = 0
        plt.style.use("CalSciPy.style")
        self.figure = plt.figure(figsize=(16, 9))
        self.figure.canvas.mpl_connect("key_press_event", self.on_key)
        self.axes = self.figure.add_subplot(111)
        self.set_labels()
        plt.show()

    def set_labels(self):
        self.axes.clear()
        self.axes.set_title(self.title)
        self.axes.set_xlabel(self.x_label)
        self.axes.set_ylabel(self.y_label)


class SpikePlot(InteractivePlot):
    def __init__(self,
                 spike_prob: np.ndarray = None,
                 spike_times: np.ndarray = None,
                 traces: np.ndarray = None,
                 frame_rate: float = None,
                 y_label: str = "Δf/f0",
                 mode="overlay"
                 ):

        self.spike_prob = spike_prob

        self.spike_times = spike_times
        if isinstance(spike_times, np.ndarray):
            self.spike_times = [np.where(spike_times[neuron, :] == 1)[0] for neuron in spike_times.shape[0]]

        self.traces = traces
        self.frame_rate = frame_rate
        self.mode = mode
        self.y_label = y_label

        if self.frame_rate:
            self.x_label = "Time (s)"
        else:
            self.x_label = "Frame (#)"
        self.time = self.set_time()

        self.title_template = f"Trace: Neuron "

        super().__init__()

        self.plot()

    @property
    def frames(self):
        if self.spike_prob is not None:
            return self.spike_prob.shape[-1]
        else:
            return self.traces.shape[-1]

    @property
    def neurons(self):
        if self.spike_prob is not None:
            return self.spike_prob.shape[0]
        else:
            return self.traces.shape[0]

    def loop(self, event):
        if event.key == "up":
            if 0 <= self.pointer + 1 <= self.neurons - 1:
                self.pointer += 1
                self.plot()
        elif event.key == "down":
            if 0 <= self.pointer - 1 <= self.neurons - 1:
                self.pointer -= 1
                self.plot()

    def plot(self):
        self.set_labels()
        if self.spike_prob is not None:
            self.axes.plot(self.time, self.spike_prob[self.pointer, :] - 1, lw=1.5, alpha=0.95, color=COLORS.red)
        if self.traces is not None:
            self.axes.plot(self.time, self.traces[self.pointer, :], lw=1.5, alpha=0.95, color=COLORS.blue)
        if self.spike_times is not None:
            for spike in self.spike_times[self.pointer]:
                a.plot(np.asarray([self.time[spike], self.time[spike]]), [-1.4, -1.2], "k")
        self.set_limits()

    def set_limits(self):
        if self.spike_prob is not None:
            self.axes.set_xlim([0, self.time[-1]])
        else:
            self.axes.set_xlim([0, self.time[-1]])

    def set_time(self):
        if self.frame_rate:
            return np.arange(0, self.frames / self.frame_rate - 1/self.frame_rate, 1 / self.frame_rate)
        else:
            return np.arange(0, self.frames, 1)

