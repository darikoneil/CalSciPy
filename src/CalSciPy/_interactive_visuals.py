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


from CalSciPy.misc import generate_time_vector


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

    def init_figure(self, one_axes=True):
        self.pointer = 0
        plt.style.use("CalSciPy.style")
        self.figure = plt.figure(figsize=(16, 9))
        self.figure.canvas.mpl_connect("key_press_event", self.on_key)
        if one_axes:
            self.axes = self.figure.add_subplot(111)
            self.set_labels()
        plt.show()

    def set_labels(self, axes=None, title=None):
        if not axes:
            axes = self.axes
        if not title:
            title = self.title

        axes.clear()
        axes.set_title(title)
        axes.set_xlabel(self.x_label)
        axes.set_ylabel(self.y_label)

    def default_labels(self, function: Callable, *args, **kwargs):
        # noinspection PyShadowingNames
        def decorator(*args, **kwargs):
            args = list(args)
            arg_vals = [self.axes, self.title, self.x_label, self.y_label]
            kwargs_keys = ["axes", "title", "x_label", "y_label"]
            for idx, arg in enumerate(args):
                if not arg:
                    args[idx] = arg_vals[idx]
            args = tuple(args)
            return function(*args, **kwargs)
        return decorator


class SpikePlot(InteractivePlot):
    def __init__(self,
                 spike_prob: np.ndarray = None,
                 spike_times: np.ndarray = None,
                 traces: np.ndarray = None,
                 frame_rate: float = None,
                 y_label: str = "Δf/f0"
                 ):

        self.spike_prob = spike_prob

        if isinstance(spike_times, np.ndarray):
            self.spike_times = [np.where(spike_times[neuron, :] == 1)[0] for neuron in range(spike_times.shape[0])]
        else:
            self.spike_times = spike_times

        self.traces = traces
        self.frame_rate = frame_rate
        self.y_label = y_label

        if self.frame_rate:
            self.x_label = "Time (s)"
        else:
            self.x_label = "Frame (#)"
        self.time = self.set_time()

        self.title_template = f"Spike Inference: Neuron "

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
                self.axes.plot(np.asarray([self.time[spike], self.time[spike]]), [-1.4, -1.2], "k")
        self.set_limits()

    def set_limits(self):
        if self.spike_prob is not None:
            self.axes.set_xlim([0, self.time[-1]])
        else:
            self.axes.set_xlim([0, self.time[-1]])

    def set_time(self):
        if self.frame_rate:
            return generate_time_vector(self.frames, self.frame_rate)
        else:
            return generate_time_vector(self.frames, step=1)


class TracePlot(InteractivePlot):
    def __init__(self,
                 traces: np.ndarray,
                 frame_rate: float = None,
                 y_label: str = "Δf/f0",
                 mode: str = "overlay"):

        self.traces = traces
        self.frame_rate = frame_rate
        self.y_label = y_label

        if isinstance(self.traces, np.ndarray):
            self.datasets = 1
        else:
            self.datasets = len(self.traces)

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
        if self.datasets == 1:
            return self.traces.shape[-1]
        else:
            return self.traces[0].shape[-1]

    @property
    def neurons(self):
        if self.datasets == 1:
            return self.traces.shape[0]
        else:
            return self.traces[0].shape[0]

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
        if self.datasets == 1:
            self.axes.plot(self.time, self.traces[self.pointer, :], lw=1.5, alpha=0.95, color=COLORS.black)
        else:
            for idx, dataset in enumerate(self.traces):
                self.axes.plot(self.time, dataset[self.pointer, :], lw=1.5, alpha=0.95, colors=COLOR(idx))
        self.set_limits()

    def set_limits(self):
        self.axes.set_xlim([0, self.time[-1]])

    def set_time(self):
        if self.frame_rate:
            return generate_time_vector(self.frames, self.frame_rate)
        else:
            return generate_time_vector(self.frames, step=1)


class TrialPlot(InteractivePlot):
    def __init__(self,
                 data: np.ndarray,
                 trials: np.ndarray,
                 trial_conditions: None,
                 bin_duration: float = None,
                 y_label: str = "Firing Rate (Hz)"):

        self.data = data
        self.trials = trials
        self.trial_conditions = trial_conditions
        self.num_trials = np.unique(self.trials).astype(int).tolist()
        self.variables, self.frames = self.data.shape
        self.bin_duration = bin_duration
        self.y_label = y_label

        if self.bin_duration:
            self.x_label = "Time (s)"
        else:
            self.x_label = "Frame (#)"
        self.time = self.set_time()

        self.title_template = f"Variable: "

        self.pointer = 0

        super().__init__()

        self.plot()

    def loop(self, event):
        if event.key == "up":
            if 0 <= self.pointer + 1 <= self.variables - 1:
                self.pointer += 1
                self.plot()
        elif event.key == "down":
            if 0 <= self.pointer - 1 <= self.variables - 1:
                self.pointer -= 1
                self.plot()

    def plot(self):
        self.figure.suptitle(self.title)
        for condition in self.conditions:
            for trial in self.trials_per_condition:
                title = f"Condition: {condition}, Trial {trial}"
                self.axes[trial, condition].plot(self.time, self.data[self.pointer, :], lw=1.5, alpha=0.95,
                                                 color=colors.black)
                self.set_labels(axes=self.axes[trial, condition], title=title)

    @property
    def conditions(self):
        if self.trial_conditions:
            return len(self.trial_conditions)
        else:
            return 1

    @property
    def trials_per_condition(self):
        if self.trials_conditions:
            return self.num_trials // self.conditions
        else:
            return self.num_trials

    def set_limits(self):
        for condition in self.conditions:
            for trial in self.trials_per_condition:
                self.axes[trial, condition].set_xlim([self.time[0], self.time[-1]])

    def set_time(self):
        if self.bin_duration:
            return generate_time_vector(self.frames / self.bin_duration, step = 1 / self.bin_duration)
        else:
            return generate_time_vector(self.frames, step=1)
