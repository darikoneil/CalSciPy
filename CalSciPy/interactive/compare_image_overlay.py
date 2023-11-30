from __future__ import annotations
from math import floor

import numpy as np
from memoization import cached

from CalSciPy.color_scheme import COLORS

import matplotlib
matplotlib.use("QtAgg")
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
import seaborn as sns


class RegistrationComparison:
    """
    Interactive Figure for comparing registration success

    """
    def __init__(self, image_0: np.ndarray, image_1: np.ndarray):

        # pre-allocate
        self.fig = None
        self.ax = None
        self.slider = None

        # defaults
        self.cmap = "Spectral_r"
        self.grid = True
        self.line_alpha = 0.5
        self.title = "Registration Comparison: Pre-Screen vs. Stimulation Blocks"

        #: np.ndarray: image 0
        self.image_0 = image_0

        #: np.ndarray: image 1
        self.image_1 = image_1

        # make sure same size
        assert self.image_0.shape == self.image_1.shape

        self.rows, self.cols = self.image_0.shape
        self.rows -= 1
        self.cols -= 1

        self.c_min = min([self.image_0.min(), self.image_1.min()])
        self.c_max = max([self.image_0.max(), self.image_1.max()])

        # initialize
        self.init_figure()
        self.init_slider()
        self.init_image()
        self.style_figure()
        self.update(self.cols // 2)

    def init_figure(self):
        plt.style.use("CalSciPy.main")
        self.fig, self.ax = plt.subplots(1, 1)
        self.fig.set_tight_layout(True)
        self.ax.set_title(self.title)
        self.ax.grid(self.grid)

    def init_image(self):
        self.ax.imshow(self.image_0, cmap=self.cmap, vmin=self.c_min, vmax=self.c_max, interpolation="bicubic")
        self.ax.vlines(self.cols // 2, 0, self.cols, lw=2, color=(*COLORS.BLACK, self.line_alpha))

    def init_slider(self):
        self.slider = Slider(ax=self.ax,
                             label="",
                             valmin=0,
                             valmax=self.cols,
                             valinit=self.cols // 2,
                             orientation="horizontal",
                             initcolor=None,
                             track_color=None,
                             handle_style={
                                 "facecolor": None,
                                 "edgecolor": None,
                                 "size": 10
                             })
        self.slider.on_changed(self.update)

    def style_figure(self):
        self.ax.margins(0)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_title(self.title)
        self.ax.grid(self.grid)

    @staticmethod
    @cached(max_size=50, order_independent=True)
    def calculate_image(image_0: np.ndarray, image_1: np.ndarray, split: int):
        composite = np.zeros_like(image_0)
        composite[:, :split] = image_0[:, :split]
        composite[:, split:] = image_1[:, split:]
        return composite

    def update(self, val):
        val = int(floor(val))
        image = self.calculate_image(self.image_0, self.image_1, val)
        self.ax.cla()
        self.ax.imshow(image, self.cmap, vmin=self.c_min, vmax=self.c_max, interpolation="bicubic")
        self.ax.vlines(val, 0, self.rows, lw=2, color=(*COLORS.BLACK, self.line_alpha))
        self.style_figure()
