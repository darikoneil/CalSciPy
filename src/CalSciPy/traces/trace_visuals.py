from __future__ import annotations
import sys

import numpy as np

import matplotlib
matplotlib.use("Qt5Agg")  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402
import seaborn as sns

from .._interactive import InteractivePlot
from ._visual import generate_time_vector  # noqa: E402
from .color_scheme import COLORS  # noqa: E402


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

        self.title_template = "Trace: Neuron "

        super().__init__()

        self.plot()

    @property
    def frames(self) -> int:
        if self.datasets == 1:
            return self.traces.shape[-1]
        else:
            return self.traces[0].shape[-1]

    @property
    def neurons(self) -> int:
        if self.datasets == 1:
            return self.traces.shape[0]
        else:
            return self.traces[0].shape[0]

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
        if self.datasets == 1:
            self.axes.plot(self.time, self.traces[self.pointer, :], lw=1.5, alpha=0.95, color=COLORS.black)
        else:
            for idx, dataset in enumerate(self.traces):
                self.axes.plot(self.time, dataset[self.pointer, :], lw=1.5, alpha=0.95, colors=COLORS(idx))
        self.set_limits()

    def set_limits(self) -> None:
        self.axes.set_xlim([0, self.time[-1]])

    def set_time(self) -> None:
        if self.frame_rate:
            return generate_time_vector(self.frames, self.frame_rate)
        else:
            return generate_time_vector(self.frames, step=1)


def plot_traces(traces: np.ndarray,
                frame_rate: float = None,
                y_label: str = "Δf/f0",
                mode: str = "overlay") -> None:
    """

    :param traces:
    :param frame_rate:
    :param y_label:
    :param mode:
    :return:
    """
    with plt.style.context("CalSciPy.main"):
        _ = TracePlot(traces, frame_rate, y_label, mode)  # noqa: F841
