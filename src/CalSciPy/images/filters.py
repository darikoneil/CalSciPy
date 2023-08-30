from __future__ import annotations
from typing import Union, Callable, Tuple, Sequence
from numbers import Number

import numpy as np

from .._calculations import generate_blocks, wrap_cupy_block

# try to import cupy else use scipy as drop-in replacement
from scipy.ndimage import gaussian_filter as _scipy_gaussian_filter
from scipy.ndimage import median_filter as _scipy_median_filter

try:
    import cupy
    from cupyx.scipy.ndimage import gaussian_filter as _gaussian_filter
    from cupyx.scipy.ndimage import median_filter as _median_filter
except ModuleNotFoundError:
    _gaussian_filter = _scipy_gaussian_filter
    _median_filter = _scipy_median_filter


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

    .. tip::

       Median filtering is particularly beneficial for denoising images with mild-to-moderate, signal-independent noise
       (e.g., the speckle that can occur on images collecting while supplying photomultiplier tubes with high-voltage).
       It tends to cause less temporal distortion than gaussian filtering as it simply replaces outliers with common
       datapoints. It also tends to induce less blurring of edges in the image (e.g., spikes, cell borders), though
       in the worst-case both filters are equally problematic.

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
                filtered_images[block, :, :] = _median_filter(filtered_images[block, :, :], mask=mask)
        except (RuntimeError, StopIteration):
            pass
    else:
        filtered_images = _median_filter(filtered_images, mask=window)

    return filtered_images
