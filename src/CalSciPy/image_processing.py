from __future__ import annotations
from typing import Callable, List, Tuple, Sequence, Optional, Union, Any
import numpy as np
from tqdm.auto import tqdm
import scipy.ndimage
import skimage.measure
import math

from .validators import validate_longest_numpy_dimension, validate_numpy_dimension_odd

try:
    import cupy
    import cupyx.scipy.ndimage
except ModuleNotFoundError or ImportError:
    pass


@validate_longest_numpy_dimension(axis=0, pos=0)
@validate_numpy_dimension_odd(odd_dimensions=(0, 1, 2), pos=1)
def blockwise_fast_filter_tiff(images: np.ndarray, mask: np.ndarray = np.ones((3, 3, 3)), 
                               block_size: int = 21000, block_buffer: int = 500) -> np.ndarray:
    """
    GPU-parallelized multidimensional median filter performed in overlapping blocks.

    Designed for use on arrays larger than the available memory capacity.

    Footprint is of the form np.ones((frames, y pixels, x pixels)) with the origin in the cente

    :param images: images stack to be filtered
    :type images: numpy.ndarray
    :param mask: mask of the median filter
    :type mask: numpy.ndarray = np.ones((3, 3, 3))
    :param block_size: the size of each block. Must fit within memory
    :type block_size: int = 21000
    :param block_buffer: the size of the overlapping region between block
    :type block_buffer: int = 500
    :return: images: numpy array (frames, y pixels, x pixels)
    :rtype: numpy.ndarray
    """
    _total_frames = images.shape[0]
    _blocks = range(0, _total_frames, block_size)
    _num_blocks = len(_blocks)
    _remainder = np.full((block_buffer, images.shape[1], images.shape[2]), 0, dtype=np.int16)

    for _block in tqdm(
            range(_num_blocks),
            total=_num_blocks,
            desc="Filtering images...",
            disable=False,
    ):
        if _block == 0:
            _remainder = images[_blocks[_block + 1] - 500:_blocks[_block + 1], :, :].copy()
            images[0:_blocks[_block + 1], :, :] = cupy.asnumpy(fast_filter_images(cupy.asarray(
                images[0:_blocks[_block + 1], :, :]), Footprint))
        elif _block == _num_blocks - 1:

            images[_blocks[_block]:_total_frames, :, :] = \
                cupy.asnumpy(fast_filter_images(
                    cupy.asarray(np.append(_remainder, images[_blocks[_block]:_total_frames, :, :],
                                           axis=0)), Footprint))[block_buffer:, :, :]
        else:
            _remainder_new = images[_blocks[_block + 1] - 500:_blocks[_block + 1], :, :].copy()
            images[_blocks[_block]:_blocks[_block + 1], :, :] = \
                cupy.asnumpy(fast_filter_images(
                    cupy.asarray(np.append(_remainder, images[_blocks[_block]:_blocks[_block + 1], :, :],
                                           axis=0)), Footprint))[block_buffer:block_size+block_buffer, :, :]
            _remainder = _remainder_new.copy()

    return images
# REFACTOR + OPTIMIZE, MAKE OUTPUT CUPY.NDARRAY
# TODO UNIT TEST


@validate_longest_numpy_dimension(axis=0, pos=0)
def clean_image_stack(images: np.ndarray, artifact_length: int = 1000, stack_sizes: int = 7000) \
        -> np.ndarray:
    """
    Function to remove initial imaging frames such that any shutter artifact is removing and the resulting tensor
    is evenly divisible by the desired stack size

    :param images: images array (frames, y pixels, x pixels)
    :type images: numpy.ndarray
    :param artifact_length: number of frames considered artifact
    :type artifact_length: int = 1000
    :param stack_sizes: number of frames per stack
    :type stack_sizes: int = 7000
    :return: images
    :rtype: numpy.ndarray
    """
    _frames = images.shape[0]
    _crop_idx = _frames % stack_sizes
    if _crop_idx >= artifact_length:
        return images[_crop_idx:, :, :]
    else:
        _frames -= artifact_length
        _crop_idx = _num_frames % stack_sizes
        return images[artifact_length + _crop_idx:, :, :]
# TODO UNIT TEST


@validate_longest_numpy_dimension(axis=0, pos=0)
@validate_numpy_dimension_odd(odd_dimensions=(0, 1, 2), pos=1)
def fast_filter_images(images: np.ndarray, mask: np.ndarray = np.ones((3, 3, 3))) -> Any:
    """
    GPU-parallelized multidimensional median filter

    mask is of the form np.ones((frames, y pixels, x pixels)) with the origin in the center

    Requires CuPy

    :param images: image stack to be filtered (frames, y pixels, x pixels)
    :type images: numpy.ndarray
    :param mask: Mask of the median filter
    :type mask: numpy.ndarray = np.ones((3, 3, 3))
    :return: filtered_image (frames, y pixels, x pixels)
    :rtype: Any
    """
    return cupyx.scipy.ndimage.median_filter(cupy.asarray(images), footprint=mask)
# TODO UNIT TEST


@validate_longest_numpy_dimension(axis=0, pos=0)
@validate_numpy_dimension_odd(odd_dimensions=(0, 1, 2), pos=1)
def filter_images(images: np.ndarray, mask: np.ndarray = np.ones((3, 3, 3))) -> np.ndarray:
    """
    Denoise a tiff stack using a multidimensional median filter

    This function simply calls

    mask is of the form np.ones((frames, y pixels, x pixels) with the origin in the center

    :param images: images stack to be filtered (frames, y pixels, x pixels)
    :type images: numpy.ndarray
    :param mask: mask of the median filter
    :type mask: numpy.ndarray = np.ones((3, 3, 3))
    :return: filtered images (frames, y pixels, x pixels)
    :rtype: numpy.ndarray
    """
    return scipy.ndimage.median_filter(images, mask=mask)
# TODO UNIT TEST


@validate_longest_numpy_dimension(axis=0, pos=0)
def grouped_z_project(images: np.ndarray, bin_size: Union[Tuple[int, int, int], int],
                      function: Callable = np.mean) -> np.ndarray:
    """
    Utilize grouped z-project to downsample data

    Downsample example function -> np.mean

    :param images: A numpy array containing a tiff stack (frames, y pixels, x pixels)
    :type images: numpy.ndarray
    :param bin_size:  size of each bin passed to downsampling function
    :type bin_size: tuple[int, int, int] or tuple[int]
    :param function: group-z projecting function
    :type function: Callable = np.mean
    :return: downsampled image (frames, y pixels, x pixels)
    :rtype: numpy.ndarray
    """
    return skimage.measure.block_reduce(images, block_size=bin_size,
                                                     func=function).astype(images.dtype)
    # cast back down from float64
# TODO UNIT TEST
