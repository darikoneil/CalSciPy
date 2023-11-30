from __future__ import annotations
from math import floor
import sys

import numpy as np
from memoization import cached

import matplotlib
matplotlib.use("QtAgg")
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import seaborn as sns

from ..color_scheme import COLORS
from .interactive import InteractivePlot


class ImageComparison(InteractivePlot):
    def __init__(self, image_0: np.ndarray,
                 image_1: np.ndarray,
                 cmap: str = "Spectral_r",
                 grid: bool = False,
                 title: str = "Image Comparison",
                 xlabel: str = "",
                 ylabel: str = "",
                 fig: Figure = None,
                 ax: Axes = None,
                 ):
        # args
        self.image_0 = image_0
        self.image_1 = image_1
        assert self.image_0.shape == self.image_1.shape, "Images must be of identical shape!"

        # optional
        self.cmap = cmap

        # derived
        self.n_rows, self.n_cols = self.image_0.shape
        self.n_rows -= 1
        self.n_cols -= 1
        self.c_min = min([self.image_0.min(), self.image_1.min()])
        self.c_max = max([self.image_0.max(), self.image_1.max()])
        self.c_step = (self.c_max - self.c_min) // 100
        self.xlim = [0, self.n_cols]
        self.ylim = [self.n_rows, 0]

        # preset
        self.lw = 2
        self.line_alpha = 0.5

        # preallocate composite image to plot
        self.composite = np.zeros_like(self.image_0)

        # preallocate slider
        self.slider = None

        super().__init__(fig=fig,
                         ax=ax,
                         grid=grid,
                         title=title,
                         xlim=self.xlim,
                         ylim=self.ylim)

    @staticmethod
    @cached(max_size=50, order_independent=True)
    def calculate_image(image_0, image_1, split):
        composite = np.zeros_like(image_0)
        composite[:, :split] = image_0[:, :split]
        composite[:, split:] = image_1[:, split:]
        return composite

    def init_pointer(self):
        self.pointer = self.n_cols // 2

    def init_interactive(self):
        self.slider = Slider(ax=self.ax,
                             label="",
                             valmin=0,
                             valmax=self.n_cols,
                             valinit=self.pointer,
                             orientation="horizontal",
                             initcolor=None,
                             track_color=None,
                             handle_style={
                                 "facecolor": None,
                                 "edgecolor": None,
                                 "size": 10,
                             })

        self.slider.on_changed(self._update)

    def plot(self):
        self.ax.imshow(self.composite, cmap=self.cmap, vmin=self.c_min, vmax=self.c_max, interpolation=None)
        self.ax.vlines(self.pointer, 0, self.n_rows, lw=self.lw, colors=(*COLORS.BLACK, self.line_alpha))

    def supplemental_style(self):
        self.ax.set_xticks([])
        self.ax.set_yticks([])

    def update(self, val):
        self.pointer = int(floor(val))
        self.composite = self.calculate_image(self.image_0, self.image_1, self.pointer)

def compare_images(image_0, image_1, cmap="Spectral_r", grid=False, title="Image Comparison"):
    int_fig = ImageComparison(image_0, image_1, cmap=cmap, grid=grid, title=title)
    return int_fig
