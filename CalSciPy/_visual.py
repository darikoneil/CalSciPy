from __future__ import annotations
from numbers import Number
from typing import Tuple, Any
import numpy as np

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.cm import get_cmap  # noqa: E402, F401
from matplotlib import pyplot as plt  # noqa: E402, F401
from matplotlib.ticker import MultipleLocator  # noqa: E402, F401
from matplotlib.patches import Polygon  # noqa: E402, F401
import seaborn as sns  # noqa: E402, F401


"""
Where all the helpers for generic visuals live
"""


class ColorSpectrum:
    """
    Class generates a linear colormap for some number of samples
    """

    def __init__(self, samples: int, colormap: str = "Spectral_r", alpha: float = None):
        self.cmap = self._generate_color_spectrum(samples, colormap, alpha)

    @staticmethod
    def _generate_color_spectrum(samples: int, colormap: str = "Spectral_r", alpha: float = None) -> Tuple:
        cmap = get_cmap(colormap)
        points = np.linspace(0, 1, samples).tolist()
        colors = tuple([cmap(point) for point in points])
        if alpha is not None:
            colors = tuple([(*color[:3], alpha) for color in colors])
        return colors

    def _generator(self) -> Tuple[float, float, float, ...]:
        for color in self.cmap:
            yield color

    def __enter__(self):  #: noqa: ANN001
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any):  #: noqa: ANN001
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
