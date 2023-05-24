from __future__ import annotations
from typing import Union
from numbers import Number
import numpy as np
from .misc import generate_blocks, wrap_cupy_block

try:
    import cupy
    import cupyx.scipy.ndimage
except ModuleNotFoundError:
    pass

DEFAULT_MASK = np.ones((3, 3, 3))


def gaussian_filter(images: np.ndarray, sigma: Union[Number, np.ndarry] = 1.0, block_size: int = None,
                    block_buffer: int = 0, in_place: bool = False) -> np.ndarray:
    """
    GPU-parallelized multidimensional gaussian filter. Optional arguments for in-place calculation. Can be calculated
    blockwise with overlapping or non-overlapping blocks.

    Designed for use on arrays larger than the available memory capacity.

    Footprint is of the form np.ones((frames, y pixels, x pixels)) with the origin in the center

    :param images: images stack to be filtered
    :type images: numpy.ndarray
    :param sigma: sigma for gaussian filter
    :type sigma: Number or numpy.ndarray
    :param block_size: the size of each block. Must fit within memory
    :type block_size: int = None
    :param block_buffer: the size of the overlapping region between block
    :type block_buffer: int = 0
    :param in_place: whether to calculate in-place
    :type in_place: bool = False
    :return: images: numpy array (frames, y pixels, x pixels)
    :rtype: numpy.ndarray
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


def median_filter(images: np.ndarray, mask: np.ndarray = DEFAULT_MASK, block_size: int = None,
                  block_buffer: int = 0, in_place: bool = False) -> np.ndarray:
    """
    GPU-parallelized multidimensional median filter. Optional arguments for in-place calculation. Can be calculated
    blockwise with overlapping or non-overlapping blocks.

    Designed for use on arrays larger than the available memory capacity.

    Footprint is of the form np.ones((frames, y pixels, x pixels)) with the origin in the center

    :param images: images stack to be filtered
    :type images: numpy.ndarray
    :param mask: mask of the median filter
    :type mask: numpy.ndarray = np.ones((3, 3, 3))
    :param block_size: the size of each block. Must fit within memory
    :type block_size: int = None
    :param block_buffer: the size of the overlapping region between block
    :type block_buffer: int = 0
    :param in_place: whether to calculate in-place
    :type in_place: bool = False
    :return: images: numpy array (frames, y pixels, x pixels)
    :rtype: numpy.ndarray
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
                filtered_images[block, :, :] = _median_filter(filtered_images[block, :, :], mask=mask)
        except (RuntimeError, StopIteration):
            pass
    else:
        filtered_images = _median_filter(filtered_images, mask=mask)

    return filtered_images


@wrap_cupy_block
def _gaussian_filter(images: cupy.ndarray, sigma: Union[Number, np.ndarray]) -> np.ndarray:
    """
    Implementation function of gaussian filter

    :param images: images to be filtered
    :type images: cupy.ndarray
    :param sigma: sigma for filter
    :type sigma: Number or numpy.ndarray
    :return: filtered numpy array
    :rtype: numpy.ndarray
    """
    return cupyx.scipy.ndimage.gaussian_filter(images, sigma=sigma)


@wrap_cupy_block
def _median_filter(images: cupy.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Implementation function of median filter

    :param images: images to be filtered
    :type images: cupy.ndarray
    :param mask: mask for filtering
    :type mask: numpy.ndarray
    :return: filtered numpy array
    :rtype: numpy.ndarray
    """
    return cupyx.scipy.ndimage.median_filter(images, footprint=mask)
