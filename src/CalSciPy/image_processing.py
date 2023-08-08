from __future__ import annotations
from typing import Union
from numbers import Number
import numpy as np
from .misc import generate_blocks, wrap_cupy_block

try:
    import cupy
    import cupyx.scipy.ndimage
    USE_GPU = True
except ModuleNotFoundError:
    import scipy.ndimage
    USE_GPU = False
finally:
    if USE_GPU:
        _gaussian_filter = _gaussian_filter_gpu
        _median_filter = _median_filter_gpu
    else:
        _gaussian_filter = _gaussian_filter_cpu
        _median_filter = _median_filter_cpu



DEFAULT_MASK = np.ones((3, 3, 3))


def gaussian_filter(images: np.ndarray, sigma: Union[Number, np.ndarry] = 1.0, block_size: int = None,
                    block_buffer: int = 0, in_place: bool = False) -> np.ndarray:
    """
    GPU-parallelized multidimensional gaussian filter. Optional arguments for in-place calculation. Can be calculated
    blockwise with overlapping or non-overlapping blocks.

    Designed for use on arrays larger than the available memory capacity.

    Footprint is of the form np.ones((frames, y pixels, x pixels)) with the origin in the center

    :param images: images stack to be filtered
    :param sigma: sigma for gaussian filter
    :param block_size: the size of each block. Must fit within memory
    :param block_buffer: the size of the overlapping region between block
    :param in_place: whether to calculate in-place
    :returns: images: numpy array (frames, y pixels, x pixels)
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
    :param mask: mask of the median filter
    :param block_size: the size of each block. Must fit within memory
    :param block_buffer: the size of the overlapping region between block
    :param in_place: whether to calculate in-place
    :returns: images: numpy array (frames, y pixels, x pixels)
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
def _gaussian_filter_gpu(images: cupy.ndarray, sigma: Union[Number, np.ndarray]) -> np.ndarray:
    """
    Implementation function of gaussian filter

    :param images: images to be filtered
    :param sigma: sigma for filter
    :returns: filtered numpy array
    """
    return cupyx.scipy.ndimage.gaussian_filter(images, sigma=sigma)


@wrap_cupy_block
def _median_filter_gpu(images: cupy.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Implementation function of median filter

    :param images: images to be filtered
    :param mask: mask for filtering
    :returns: filtered numpy array
    """
    return cupyx.scipy.ndimage.median_filter(images, footprint=mask)


def _gaussian_filter_cpu(images: np.ndarray, sigma: Union[Number, np.ndarray]) -> np.ndarray:
    """
    Implementation function of gaussian filter

    :param images: images to be filtered
    :param sigma: sigma for filter
    :returns: filtered numpy array
    """
    return scipy.ndimage.gaussian_filter(images, sigma=sigma)


def _median_filter_cpu(images: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Implementation function of median filter

    :param images: images to be filtered
    :param mask: mask for filtering
    :returns: filtered numpy array
    """
    return scipy.ndimage.median_filter(images, footprint=mask)