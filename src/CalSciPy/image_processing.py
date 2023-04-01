from __future__ import annotations
from typing import Callable, Tuple, Union, Any
import numpy as np
from tqdm.auto import tqdm
import skimage.measure

from CalSciPy.misc import generate_blocks

try:
    import cupy
    import cupyx.scipy.ndimage
except ModuleNotFoundError:
    pass


DEFAULT_MASK = np.ones((3, 3, 3))


def median_filter(images: np.ndarray, mask: np.ndarray = DEFAULT_MASK, block_size: int = 21000,
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
    :return: images: numpy array (frames, y pixels, x pixels)
    :rtype: numpy.ndarray
    """

    if in_place:
        filtered_images = images
    else:
        filtered_images = images.copy()

    filtered_images = cupy.asarray(filtered_images)

    frames = filtered_images.shape[0]

    if block_size:
        block_gen = generate_blocks(range(frames), block_size, block_buffer)
        for block in block_gen:
            filtered_images[block, :, :] = cupyx.scipy.ndimage.median_filter(filtered_images[block, :, :],
                                                                             footprint=mask)
    else:
        filtered_images = cupyx.scipy.ndimage.median_filter(filtered_images, footprint=mask)

    return filtered_images.get()


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
    return skimage.measure.block_reduce(images, block_size=bin_size, func=function).astype(images.dtype)
    # cast back down from float64
# TODO UNIT TEST
