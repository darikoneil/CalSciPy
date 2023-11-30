from __future__ import annotations
from typing import Optional
from abc import ABC, abstractmethod

import matplotlib
matplotlib.use("QtAgg")
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.axes import Axes


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

    def init_figure(self):
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
    def plot(self):
        ...

    @abstractmethod
    def init_pointer(self):
        ...

    @abstractmethod
    def init_interactive(self):
        ...

    def style_axes(self):
        self.ax.grid(self.grid)
        self.ax.set_title(self.title)
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_xlim(self.xlim)
        self.ax.set_ylabel(self.ylabel)
        self.ax.set_ylim(self.ylim)
        self.supplemental_style()

    def supplemental_style(self):
        ...

    @abstractmethod
    def update(self, val):
        ...

    def _update(self, val):
        self._pre_update()
        self.update(val)
        self.plot()
        self._post_update()

    def _pre_update(self):
        self.ax.cla()

    def _post_update(self):
        self.style_axes()
