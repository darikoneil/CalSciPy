from __future__ import annotations
from typing import Tuple, Optional, Mapping

import numpy as np

from CalSciPy.color_scheme import COLORS

import matplotlib
matplotlib.use("QtAgg")
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import MultipleLocator  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402
import seaborn as sns


"""
ROI-Centric Plots
"""


def plot_rois(rois: Mapping,
              background: Union[np.ndarray, Tuple[int, int]],
              title: str = "ROIs",
              theme: str = "con_image",
              colormap: str = "binary",
              ax: Optional[plt.Axes] = None):

    if ax is None:
        fig, ax = _init_figure(theme)

    # generate a background image to plot over
    if isinstance(background, tuple):
        reference_shape = background
        background = np.ones((*background, 3), dtype=np.uint8) * COLORS.BACKGROUND
        vi = 0
        vm = 255
    else:
        reference_shape = background.shape
        vi, vm = _reference_image_cutoffs(background)

    # plot reference_image
    ax.imshow(background, vmin=vi, vmax=vm, cmap=colormap, interpolation="bicubic")
    ax.set_xticks([])
    ax.set_yticks([])


def _init_figure(theme: str = "con_image"):
    theme = f"CalSciPy.{theme}"
    with plt.style.context(theme):
        fig, ax = plt.subplots(1, 1)
        return fig, ax


def _reference_image_cutoffs(reference_image):
    flattened_reference = np.ravel(reference_image)
    vi = np.percentile(flattened_reference, 1)
    vm = np.percentile(flattened_reference, 99)
    vm += (3 * abs(vm))
    return vi, vm
