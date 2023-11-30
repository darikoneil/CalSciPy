from __future__ import annotations
from typing import Optional, Sequence, Any
from abc import ABC, abstractmethod

import matplotlib  # noqa: E402
matplotlib.use("QtAgg")  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: F401, E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402


class InteractivePlot(ABC):
    def __init__(self,
                 fig: Optional[Figure] = None,
                 ax: Optional[Axes] = None,
                 legend: bool = False,
                 grid: bool = True,
                 title: str = "",
                 xlabel: str = "",
                 xlim: Sequence = (0, 0),
                 ylabel: str = "",
                 ylim: Sequence = (0, 0)
                 ):

        # plot destination
        self.fig = fig
        self.ax = ax

        # default properties
        self.grid = grid
        self.title = title
        self.xlabel = xlabel
        self.xlim = xlim
        self.ylabel = ylabel
        self.ylim = ylim

        # pointer
        self.pointer = 0

        self.init_figure()

    def init_figure(self) -> "InteractivePlot":
        # create if necessary
        if self.fig is None and self.ax is None:
            with plt.style.context("CalSciPy.main"):
                self.fig, self.ax = plt.subplots(1, 1)
        # style the axes
        self.style_axes()
        self.init_pointer()
        self.init_interactive()
        self._update(self.pointer)
        self.fig.tight_layout()

    @abstractmethod
    def plot(self) -> "InteractivePlot":
        ...

    @abstractmethod
    def init_pointer(self) -> "InteractivePlot":
        ...

    @abstractmethod
    def init_interactive(self) -> "InteractivePlot":
        ...

    def style_axes(self) -> "InteractivePlot":
        self.ax.grid(self.grid)
        self.ax.set_title(self.title)
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_xlim(self.xlim)
        self.ax.set_ylabel(self.ylabel)
        self.ax.set_ylim(self.ylim)
        self.supplemental_style()

    def supplemental_style(self) -> "InteractivePlot":  # noqa: B027
        # optional method to add some more style
        ...

    @abstractmethod
    def update(self, val: Any) -> "InteractivePlot":
        ...

    def _update(self, val: Any) -> "InteractivePlot":
        self._pre_update()
        self.update(val)
        self.plot()
        self._post_update()

    def _pre_update(self) -> "InteractivePlot":
        self.ax.cla()

    def _post_update(self) -> "InteractivePlot":
        self.style_axes()
