from __future__ import annotations
import numpy as np
from ._interactive_visuals import SpikePlot, TracePlot, TrialPlot

import matplotlib
matplotlib.use("Qt5Agg")  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402


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
