from __future__ import annotations
from typing import Union, Tuple, Sequence
from numbers import Number

import numpy as np

from .._calculations import generate_blocks

# try to import cupy else use scipy as drop-in replacement
from scipy.ndimage import gaussian_filter as _scipy_gaussian_filter
from scipy.ndimage import median_filter as _scipy_median_filter
from scipy.ndimage import uniform_filter as _scipy_mean_filter

try:
    from cupyx.scipy.ndimage import gaussian_filter as _gaussian_filter
    from cupyx.scipy.ndimage import median_filter as _median_filter
    from cupyx.scipy.ndimage import uniform_filter as _mean_filter
    from .._calculations import wrap_cupy_block
    _gaussian_filter = wrap_cupy_block(_gaussian_filter)
    _median_filter = wrap_cupy_block(_median_filter)
    _mean_filter = wrap_cupy_block(_median_filter)
    USE_GPU = True
except ModuleNotFoundError:
    _gaussian_filter = _scipy_gaussian_filter
    _median_filter = _scipy_median_filter
    _mean_filter = _scipy_mean_filter
    USE_GPU = False


"""
Collection of filters for denoising images. Standard implementation is CuPy, reverts to SciPy if unavailable.
"""


def gaussian_filter(images: np.ndarray,
                    sigma: Union[Number, np.ndarry, Sequence[Number]] = 1.0,
                    block_size: int = None,
                    block_buffer: int = 0,
                    in_place: bool = False) -> np.ndarray:
    """
    Multidimensional Gaussian filter with GPU-implementation through `CuPy <https://cupy.dev>`_\.
    If `CuPy <https://cupy.dev>`_ is not installed `SciPy <https://scipy.org>`_ is used as a (slower) replacement.

    :param images: Images to be filtered (frames, y-pixels, x-pixels)

    :type images: :class:`ndarray <numpy.ndarray>`

    :param sigma: Standard deviation for Gaussian kernel

    :param block_size: The number of frames in each processing block.

    :param block_buffer: The size of the overlapping region between block

    :param in_place: Whether to calculate in-place

    :returns: Filtered images (frames, y pixels, x pixels)

    :rtype: :class:`ndarray <numpy.ndarray>`

    .. warning::

        The number of frames in each processing block must fit within your GPU's VRAM (`CuPy <https://cupy.dev>`_)
        or RAM (`SciPy <https://scipy.org>`_). This function will not automatically revert to the SciPy implementation
        if there is not sufficient VRAM. Instead, an out of memory error will be raised.

    """
    if in_place:
        filtered_images = images
    else:
        filtered_images = images.copy()

    frames = filtered_images.shape[0]

    if block_size:
        block_gen = generate_blocks(range(frames), block_size, block_buffer)
        try:
            for block in block_gen:
                filtered_images[block, :, :] = _gaussian_filter(filtered_images[block, :, :], sigma=sigma)
        except (RuntimeError, StopIteration):
            pass
    else:
        filtered_images = _gaussian_filter(filtered_images, sigma=sigma)

    return filtered_images


def mean_filter(images: np.ndarray,
                window: Tuple[int, int, int] = (1, 0, 0),
                block_size: int = None,
                block_buffer: int = 0,
                in_place: bool = False) -> np.ndarray:
    """
    Multidimensional mean (uniform) filter with GPU-implementation through `CuPy <https://cupy.dev>`_\.
    If `CuPy <https://cupy.dev>`_ is not installed, `SciPy <https://scipy.org>`_ is used as a (slower) replacement.

    :param images: Images to be filtered (frames, y-pixels, x-pixels)

    :type images: :class:`ndarray <numpy.ndarray>`

    :param window: Window of the mean filter

    :param block_size: The number of frames in each processing block.

    :param block_buffer: The size of the overlapping region between block.

    :param in_place: Whether to calculate in-place

    :returns: Filtered images (frames, y pixels, x pixels)

    :rtype: :class:`ndarray <numpy.ndarray>`

    .. versionadded:: 0.8.0

    .. warning:: Currently untested

    .. warning::

        The number of frames in each processing block must fit within your GPU's VRAM (`CuPy <https://cupy.dev>`_)
        or RAM (`SciPy <https://scipy.org>`_). This function will not automatically revert to the SciPy implementation
        if there is not sufficient VRAM. Instead, an out of memory error will be raised.

    """

    window = np.ones(window)

    if in_place:
        filtered_images = images
    else:
        filtered_images = images.copy()

    frames = filtered_images.shape[0]

    if block_size:
        block_gen = generate_blocks(range(frames), block_size, block_buffer)
        try:
            for block in block_gen:
                filtered_images[block, :, :] = _mean_filter(filtered_images[block, :, :], size=window)
        except (RuntimeError, StopIteration):
            pass
    else:
        filtered_images = _mean_filter(filtered_images, size=window)

    return filtered_images


def median_filter(images: np.ndarray,
                  window: Tuple[int, int, int] = (3, 3, 3),
                  block_size: int = None,
                  block_buffer: int = 0,
                  in_place: bool = False) -> np.ndarray:
    """
    Multidimensional median filter with GPU-implementation through `CuPy <https://cupy.dev>`_\.
    If `CuPy <https://cupy.dev>`_ is not installed, `SciPy <https://scipy.org>`_ is used as a (slower) replacement.

    :param images: Images to be filtered (frames, y-pixels, x-pixels)

    :type images: :class:`ndarray <numpy.ndarray>`

    :param window: Window of the median filter

    :param block_size: The number of frames in each processing block.

    :param block_buffer: The size of the overlapping region between block.

    :param in_place: Whether to calculate in-place

    :returns: Filtered images (frames, y pixels, x pixels)

    :rtype: :class:`ndarray <numpy.ndarray>`

    .. warning::

        The number of frames in each processing block must fit within your GPU's VRAM (`CuPy <https://cupy.dev>`_)
        or RAM (`SciPy <https://scipy.org>`_). This function will not automatically revert to the SciPy implementation
        if there is not sufficient VRAM. Instead, an out of memory error will be raised.

    """

    window = np.ones(window)

    if in_place:
        filtered_images = images
    else:
        filtered_images = images.copy()

    frames = filtered_images.shape[0]

    if block_size:
        block_gen = generate_blocks(range(frames), block_size, block_buffer)
        try:
            for block in block_gen:
                filtered_images[block, :, :] = _median_filter(filtered_images[block, :, :], footprint=window)
        except (RuntimeError, StopIteration):
            pass
    else:
        filtered_images = _median_filter(filtered_images, footprint=window)

    return filtered_images
