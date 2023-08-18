from __future__ import annotations
from typing import Union, Callable, Tuple
from numbers import Number
import numpy as np

from .._calculations import generate_blocks, wrap_cupy_block

import cupy  # noqa: F401
import cupyx.scipy.ndimage  # noqa: F401


@wrap_cupy_block
def _gaussian_filter_gpu(images: cupy.ndarray, sigma: Union[Number, np.ndarray]) -> np.ndarray:
    """
    Implementation function of gaussian filter

    :param images: images to be filtered
    :param sigma: sigma for filter
    :returns: filtered numpy array
    """
    return cupyx.scipy.ndimage.gaussian_filter(images, sigma=sigma)  # noqa: F821


@wrap_cupy_block
def _median_filter_gpu(images: cupy.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Implementation function of median filter

    :param images: images to be filtered
    :param mask: mask for filtering
    :returns: filtered numpy array
    """
    return cupyx.scipy.ndimage.median_filter(images, footprint=mask)  # noqa: F821

