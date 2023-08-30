from __future__ import annotations
from typing import Any
import numpy as np
from .._interactive import InteractivePlot

import matplotlib
matplotlib.use("Qt5Agg")  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402
import seaborn as sns

from .._visual import generate_time_vector  # noqa: E402
from ..color_scheme import COLORS  # noqa: E402


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

        self.title_template = "Spike Inference: Neuron "

        super().__init__()

        self.plot()

    @property
    def frames(self) -> int:
        if self.spike_prob is not None:
            return self.spike_prob.shape[-1]
        else:
            return self.traces.shape[-1]

    @property
    def neurons(self) -> int:
        if self.spike_prob is not None:
            return self.spike_prob.shape[0]
        else:
            return self.traces.shape[0]

    def loop(self, event: Any) -> None:
        if event.key == "up":
            if 0 <= self.pointer + 1 <= self.neurons - 1:
                self.pointer += 1
                self.plot()
        elif event.key == "down":
            if 0 <= self.pointer - 1 <= self.neurons - 1:
                self.pointer -= 1
                self.plot()

    def plot(self) -> None:
        self.set_labels()
        if self.spike_prob is not None:
            self.axes.plot(self.time, self.spike_prob[self.pointer, :] - 1, lw=1.5, alpha=0.95, color=COLORS.RED)
        if self.traces is not None:
            self.axes.plot(self.time, self.traces[self.pointer, :], lw=1.5, alpha=0.95, color=COLORS.BLUE)
        if self.spike_times is not None:
            for spike in self.spike_times[self.pointer]:
                self.axes.plot(np.asarray([self.time[spike], self.time[spike]]), [-1.4, -1.2], "k")
        self.set_limits()

    def set_limits(self) -> None:
        if self.spike_prob is not None:
            self.axes.set_xlim([0, self.time[-1]])
        else:
            self.axes.set_xlim([0, self.time[-1]])

    def set_time(self) -> None:
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

        self.title_template = "Variable: "

        self.pointer = 0

        super().__init__()

        self.plot()

    @property
    def conditions(self) -> int:
        if self.trial_conditions:
            return len(self.trial_conditions)
        else:
            return 1

    @property
    def trials_per_condition(self) -> int:
        if self.trials_conditions:
            return self.num_trials // self.conditions
        else:
            return self.num_trials

    def loop(self, event: Any) -> None:
        if event.key == "up":
            if 0 <= self.pointer + 1 <= self.variables - 1:
                self.pointer += 1
                self.plot()
        elif event.key == "down":
            if 0 <= self.pointer - 1 <= self.variables - 1:
                self.pointer -= 1
                self.plot()

    def plot(self) -> None:
        self.figure.suptitle(self.title)
        for condition in self.conditions:
            for trial in self.trials_per_condition:
                title = f"Condition: {condition}, Trial {trial}"
                self.axes[trial, condition].plot(self.time, self.data[self.pointer, :], lw=1.5, alpha=0.95,
                                                 color=COLORS.black)
                self.set_labels(axes=self.axes[trial, condition], title=title)

    def set_limits(self) -> None:
        for condition in self.conditions:
            for trial in self.trials_per_condition:
                self.axes[trial, condition].set_xlim([self.time[0], self.time[-1]])

    def set_time(self) -> None:
        if self.bin_duration:
            return generate_time_vector(self.frames / self.bin_duration, step=1 / self.bin_duration)
        else:
            return generate_time_vector(self.frames, step=1)


def plot_spikes(spike_prob: np.ndarray = None,
                spike_times: np.ndarray = None,
                traces: np.ndarray = None,
                frame_rate: float = None,
                y_label: str = "Δf/f0") -> None:
    """
    Function to interactively visualize spike inference

    :param spike_prob:
    :param spike_times:
    :param traces:
    :param frame_rate:
    :param y_label:
    """
    with plt.style.context("CalSciPy.main"):
        _ = SpikePlot(spike_prob, spike_times, traces, frame_rate, y_label)  # noqa: F841


def plot_trials(data: np.ndarray,
                trials: np.ndarray,
                trial_conditions: None,
                bin_duration: float = None,
                y_label: str = "Firing Rate (Hz)") -> None:
    """

    :param data:
    :param trials:
    :param trial_conditions:
    :param bin_duration:
    :param y_label:
    :return:
    """
    with plt.style.context("CalSciPy.main"):
        _ = TrialPlot(data, trials, trial_conditions, bin_duration, y_label)  # noqa: F841
