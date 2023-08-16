from __future__ import annotations
from typing import Callable, Any
import sys
from abc import ABC, abstractmethod
import numpy as np

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib import pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: F401, E402

from ._visual import generate_time_vector  # noqa: E402
from .color_scheme import COLORS  # noqa: E402


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
    def title(self) -> str:
        return self.title_template + f"{self.pointer}"

    @abstractmethod
    def loop(self, event: Any) -> None:
        ...

    @abstractmethod
    def set_limits(self) -> None:
        ...

    def default_labels(self, function: Callable, *args, **kwargs) -> Callable:
        # noinspection PyShadowingNames
        def decorator(*args, **kwargs) -> Callable:
            args = list(args)
            arg_vals = [self.axes, self.title, self.x_label, self.y_label]
            kwargs_keys = ["axes", "title", "x_label", "y_label"]  # noqa: F841
            for idx, arg in enumerate(args):
                if not arg:
                    args[idx] = arg_vals[idx]
            args = tuple(args)
            return function(*args, **kwargs)
        return decorator

    def init_figure(self, one_axes: bool = True) -> None:
        self.pointer = 0
        plt.style.use("CalSciPy.style")
        self.figure = plt.figure(figsize=(16, 9))
        self.figure.canvas.mpl_connect("key_press_event", self.on_key)
        if one_axes:
            self.axes = self.figure.add_subplot(111)
            self.set_labels()
        plt.show()

    def on_key(self, event: Any) -> None:
        sys.stdout.flush()
        self.loop(event)
        self.figure.canvas.draw()

    def set_labels(self, axes: Any = None, title: str = None) -> None:
        if not axes:
            axes = self.axes
        if not title:
            title = self.title

        axes.clear()
        axes.set_title(title)
        axes.set_xlabel(self.x_label)
        axes.set_ylabel(self.y_label)
