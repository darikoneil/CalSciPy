from __future__ import annotations
from typing import Callable, Tuple, Optional, Union, Sequence
import numpy as np

from numpy.lib.stride_tricks import as_strided


"""
Projecting Imaging Stacks
"""


def view_as_blocks(arr_in, block_shape):
    if not isinstance(block_shape, tuple):
        raise TypeError('block needs to be a tuple')

    block_shape = np.array(block_shape)
    if (block_shape <= 0).any():
        raise ValueError("'block_shape' elements must be strictly positive")

    if block_shape.size != arr_in.ndim:
        raise ValueError("'block_shape' must have the same length "
                         "as 'arr_in.shape'")

    arr_shape = np.array(arr_in.shape)
    if (arr_shape % block_shape).sum() != 0:
        raise ValueError("'block_shape' is not compatible with 'arr_in'")

    # -- restride the array to build the block view
    new_shape = tuple(arr_shape // block_shape) + tuple(block_shape)
    new_strides = tuple(arr_in.strides * block_shape) + arr_in.strides

    arr_out = as_strided(arr_in, shape=new_shape, strides=new_strides)

    return arr_out


def grouped_z_project(images: np.ndarray,
                      group_size: Union[Sequence[int, ...], int],
                      func: Optional[Callable] = None
                      ) -> np.ndarray:

    if func is None:
        func = np.mean

    if not isinstance(group_size, Sequence):
        group_size = (group_size, ) * images.ndim

    while len(group_size) != images.ndim:
        group_size = (*group_size, group_size[-1])

    assert (0 not in group_size)

    pad_widths = []
    # make blocks
    for dim, gs in enumerate(group_size):
        if images.shape[dim] % gs != 0:
            pad = gs - (images.shape[dim] % gs)
        else:
            pad = 0
        pad_widths.append((0, pad))

    if np.any(np.asarray(pad_widths)):
        images = np.pad(images, pad_width=pad_widths, mode='constant', constant_values=0)

    blocked = view_as_blocks(images, group_size)

    return func(blocked, axis=tuple(range(images.ndim, blocked.ndim)))
