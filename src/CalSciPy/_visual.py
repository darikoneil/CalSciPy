from __future__ import annotations
from numbers import Number

import numpy as np

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.cm import get_cmap  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402
import seaborn as sns  # noqa: E402


"""
Where all the helpers for visuals live
"""


class ColorSpectrum:
    """
    Class generates a linear colormap for some number of samples
    """

    def __init__(self, samples: int, colormap: str = "Spectral_r", alpha: float = None):
        self.cmap = self._generate_color_spectrum(samples, colormap, alpha)

    @staticmethod
    def _generate_color_spectrum(samples: int, colormap: str = "Spectral_r", alpha: float = None):
        cmap = get_cmap(colormap)
        points = np.linspace(0, 1, samples).tolist()
        colors = tuple([cmap(point) for point in points])
        if alpha is not None:
            colors = tuple([(*color[:3], alpha) for color in colors])
        return colors

    def _generator(self):
        for color in self.cmap:
            yield color

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        ...


def generate_time_vector(num_samples: int,
                         sampling_frequency: Number = 30.0,
                         start: Number = 0.0,
                         step: Number = None
                         ) -> np.ndarray:
    """
    Generates a time vector for a number of samples collected at either

    """

    if not step:
        step = 1 / sampling_frequency

    return np.arange(0, num_samples) * step + start
