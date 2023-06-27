from __future__ import annotations
import numpy as np
from typing import Any
import sys
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt   # noqa: E402
from matplotlib.widgets import Slider  # noqa: E402
import seaborn as sns  # noqa: F401, E402


def interactive_traces(traces: np.ndarray, frame_rate: float, **kwargs) -> None:
    """
    Function to interactive compare traces. Press Up/Down to switch neurons

    :param traces: primary traces
    :param frame_rate: frame rate
    :returns: interactive figure
    """
    _num_neurons, _num_frames = traces.shape

    _line_width = kwargs.get("lw", 3)
    _alpha = kwargs.get("alpha", 0.95)

    x = np.arange(0, (_num_frames * (1 / frame_rate)), 1 / frame_rate, dtype=np.float64)

    fig = plt.figure(figsize=(16, 12))
    ax1 = fig.add_subplot(111)
    ax1.set_title("Trace: Neuron 1")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Signal (a.u.)")
    ax1.plot(x, traces[0, :], color="#40cc8b", lw=_line_width, alpha=_alpha)
    ax1.set_xlim([0, x[-1]])
    fig.subplots_adjust(bottom=0.25, hspace=0.5)
    xmin = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    xmin_slider = Slider(
        ax=xmin,
        label="X-Min",
        valmin=0,
        valmax=x[-1],
        valinit=0,
        valstep=(1 / frame_rate),
        color="#7840cc"
    )

    xmax = fig.add_axes([0.25, 0.05, 0.65, 0.03])
    xmax_slider = Slider(
        ax=xmax,
        label="X-Max",
        valmin=0,
        valmax=x[-1],
        valinit=x[-1],
        valstep=(1 / frame_rate),
        color="#7840cc"
    )

    def update(val: Any) -> None:  # noqa: U100
        ax1.set_xlim([xmin_slider.val, xmax_slider.val])
        fig.canvas.draw_idle()

    xmin_slider.on_changed(update)
    xmax_slider.on_changed(update)

    def on_key(event: Any) -> None:
        sys.stdout.flush()
        # print("press", event.key)
        if event.key == "up":
            # print('press', event.key)
            n = int(ax1.title._text.split()[2]) - 1  # ZERO INDEXED
            if n < traces.shape[0] - 1:
                n += 1  # new n
                ax1.clear()

                ax1.set_xlabel("Time (s)")
                ax1.set_ylabel("Signal (a.u.)")
                ax1.plot(x, traces[n, :], color="#40cc8b", lw=_line_width, alpha=_alpha)
                ax1.set_xlim([0, x[-1]])
                n += 1
                ax1.set_title("Trace: Neuron " + str(n))
                fig.canvas.draw()

        elif event.key == "down":
            # print('press', event.key)
            n = int(ax1.title._text.split()[2]) - 1  # ZERO INDEXED
            if n > 0:
                n -= 1  # new n
                ax1.clear()

                ax1.set_xlabel("Time (s)")
                ax1.set_ylabel("Signal (a.u.)")
                ax1.plot(x, traces[n, :], color="#40cc8b", lw=_line_width, alpha=_alpha)

                n += 1
                ax1.set_title("Trace: Neuron " + str(n))
                ax1.set_xlim([0, x[-1]])
                fig.canvas.draw()

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()


def interactive_traces_overlay(traces: np.ndarray, traces2: np.ndarray, frame_rate: float, **kwargs) -> None:
    """
    Function to interactive compare traces with an overlay trace (e.g., noise). Press Up/Down to switch neurons

    :param traces: primary traces
    :param traces2: secondary trace
    :param frame_rate: frame_rate
    :returns: interactive figure
    """
    _num_neurons, _num_frames = traces.shape

    _line_width = kwargs.get("lw", 3)
    _alpha = kwargs.get("alpha", 0.95)

    x = np.arange(0, (_num_frames * (1 / frame_rate)), 1 / frame_rate, dtype=np.float64)

    fig = plt.figure(figsize=(16, 12))
    ax1 = fig.add_subplot(111)
    ax1.set_title("Trace: Neuron 1")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Signal (a.u.)")
    ax1.plot(x, traces2[0, :], color="#40cc8b", lw=_line_width, alpha=0.25)
    ax1.plot(x, traces[0, :], color="#40cc8b", lw=_line_width, alpha=_alpha)
    ax1.set_xlim([0, x[-1]])
    fig.subplots_adjust(bottom=0.25, hspace=0.5)
    xmin = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    xmin_slider = Slider(
        ax=xmin,
        label="X-Min",
        valmin=0,
        valmax=x[-1],
        valinit=0,
        valstep=(1 / frame_rate),
        color="#7840cc"
    )

    xmax = fig.add_axes([0.25, 0.05, 0.65, 0.03])
    xmax_slider = Slider(
        ax=xmax,
        label="X-Max",
        valmin=0,
        valmax=x[-1],
        valinit=x[-1],
        valstep=(1 / frame_rate),
        color="#7840cc"
    )

    def update(val: Any) -> None:  # noqa: U100
        ax1.set_xlim([xmin_slider.val, xmax_slider.val])
        fig.canvas.draw_idle()

    xmin_slider.on_changed(update)
    xmax_slider.on_changed(update)

    def on_key(event: Any) -> None:
        sys.stdout.flush()
        # print("press", event.key)
        if event.key == "up":
            # print('press', event.key)
            n = int(ax1.title._text.split()[2]) - 1  # ZERO INDEXED
            if n < traces.shape[0] - 1:
                n += 1  # new n
                ax1.clear()

                ax1.set_xlabel("Time (s)")
                ax1.set_ylabel("Signal (a.u.)")
                ax1.plot(x, traces2[n, :], color="#40cc8b", lw=_line_width, alpha=0.25)
                ax1.plot(x, traces[n, :], color="#40cc8b", lw=_line_width, alpha=_alpha)
                ax1.set_xlim([0, x[-1]])
                n += 1
                ax1.set_title("Trace: Neuron " + str(n))
                fig.canvas.draw()

        elif event.key == "down":
            # print('press', event.key)
            n = int(ax1.title._text.split()[2]) - 1  # ZERO INDEXED
            if n > 0:
                n -= 1  # new n
                ax1.clear()

                ax1.set_xlabel("Time (s)")
                ax1.set_ylabel("Signal (a.u.)")
                ax1.plot(x, traces2[n, :], color="#40cc8b", lw=_line_width, alpha=0.25)
                ax1.plot(x, traces[n, :], color="#40cc8b", lw=_line_width, alpha=_alpha)

                n += 1
                ax1.set_title("Trace: Neuron " + str(n))
                ax1.set_xlim([0, x[-1]])
                fig.canvas.draw()

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()


def interactive_traces_compare(traces, frame_rate: float, colors, **kwargs) -> None:
    """
    Function to interactively compare two sets of traces. Press Up/Down to switch neurons

    :param traces: primary traces
    :param traces2: secondary trace
    :param frame_rate: frame_rate
    :returns: interactive figure
    """
    _num_neurons, _num_frames = traces[0].shape
    _line_width = kwargs.get("lw", 1.5)
    _alpha = kwargs.get("alpha", 0.95)

    x = np.arange(0, (_num_frames * (1 / frame_rate)), 1 / frame_rate, dtype=np.float64)

    fig = plt.figure(figsize=(16, 12))
    ax1 = fig.add_subplot(111)
    ax1.set_title("Trace: Neuron 1")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Signal (a.u.)")
    for idx, trace in enumerate(traces):
        ax1.plot(x, trace[0, :], color=colors[idx], lw=_line_width, alpha=_alpha)
    ax1.set_xlim([0, x[-1]])
    fig.subplots_adjust(bottom=0.25, hspace=0.5)
    xmin = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    xmin_slider = Slider(
        ax=xmin,
        label="X-Min",
        valmin=0,
        valmax=x[-1],
        valinit=0,
        valstep=(1 / frame_rate),
        color="#7840cc"
    )

    xmax = fig.add_axes([0.25, 0.05, 0.65, 0.03])
    xmax_slider = Slider(
        ax=xmax,
        label="X-Max",
        valmin=0,
        valmax=x[-1],
        valinit=x[-1],
        valstep=(1 / frame_rate),
        color="#7840cc"
    )

    def update(val: Any) -> None:  # noqa: U100
        ax1.set_xlim([xmin_slider.val, xmax_slider.val])
        fig.canvas.draw_idle()

    xmin_slider.on_changed(update)
    xmax_slider.on_changed(update)

    def on_key(event: Any) -> None:
        sys.stdout.flush()
        # print("press", event.key)
        if event.key == "up":
            # print('press', event.key)
            n = int(ax1.title._text.split()[2]) - 1  # ZERO INDEXED
            if n < traces[0].shape[0] - 1:
                n += 1  # new n
                ax1.clear()

                ax1.set_xlabel("Time (s)")
                ax1.set_ylabel("Signal (a.u.)")
                for idx, trace in enumerate(traces):
                    ax1.plot(x, trace[n, :], color=colors[idx], lw=_line_width, alpha=_alpha)
                ax1.set_xlim([0, x[-1]])
                n += 1
                ax1.set_title("Trace: Neuron " + str(n))
                fig.canvas.draw()

        elif event.key == "down":
            # print('press', event.key)
            n = int(ax1.title._text.split()[2]) - 1  # ZERO INDEXED
            if n > 0:
                n -= 1  # new n
                ax1.clear()

                ax1.set_xlabel("Time (s)")
                ax1.set_ylabel("Signal (a.u.)")
                for idx, trace in enumerate(traces):
                    ax1.plot(x, trace[n, :], color=colors[idx], lw=_line_width, alpha=_alpha)
                n += 1
                ax1.set_title("Trace: Neuron " + str(n))
                ax1.set_xlim([0, x[-1]])
                fig.canvas.draw()

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()


def interactive_spikes(spike_prob, spike_)