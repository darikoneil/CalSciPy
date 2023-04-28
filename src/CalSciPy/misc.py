from __future__ import annotations
from typing import Iterable, Iterator, Any, Callable
from collections import deque
from pathlib import Path
from numbers import Number
from functools import wraps

import numpy as np

try:
    import cupy
except ModuleNotFoundError:
    pass


def calculate_frames_per_file(y_pixels: int, x_pixels: int, bit_depth: np.dtype = np.uint16, size_cap: Number = 3.9) \
        -> int:
    """
    Estimates the number of image frames to allocate to each file given some maximum size.

    :param y_pixels: number of y_pixels in image
    :type y_pixels: int
    :param x_pixels: number of x_pixels in image
    :type x_pixels: int
    :param bit_depth: bit-depth / type of image elements
    :type bit_depth: numpy.dtype = numpy.uint16
    :param size_cap: maximum file size
    :type size_cap: float = 3.9
    :return: the maximum number of frames to allocate for each file
    :rtype: int
    """
    single_frame_size = np.ones((y_pixels, x_pixels), dtype=bit_depth).nbytes * 1e-9
    return size_cap // single_frame_size


def generate_blocks(sequence: Iterable, block_size: int, block_buffer: int = 0) -> Iterator:
    """
    Returns a generator of some arbitrary iterable sequence that yields m blocks with overlapping regions of size n

    :param sequence: Sequence to be split into overlapping blocks
    :type sequence: Iterable
    :param block_size: size of blocks
    :type block_size: int
    :param block_buffer: size of overlap between blocks
    :type block_buffer: int
    :return: generator yielding m blocks with overlapping regions of size n
    :rtype: Iterator
    """
    if block_size <= 1:
        raise ValueError("Block size must be > 1")
    if block_buffer >= block_size:
        raise AssertionError("Block buffer must be smaller than the size of the block.")
    if block_size >= len(sequence):
        raise AssertionError("Block must be smaller than iterable sequence")

    block_size = int(block_size)  # coerce in case float

    block = deque(maxlen=block_size)
    iterable = iter(sequence)

    # make first block
    for _ in range(block_size):
        block.append(next(iterable))

    while True:
        try:
            yield tuple(block)
            for idx in range(block_size - block_buffer):  # noqa: B007
                block.append(next(iterable))  # we subtract the block buffer to make space for overlap

        except StopIteration:
            # noinspection PyUnboundLocalVariable
            idx += block_buffer  # make sure that we always have at least the overlap
            if idx == 0:
                pass  # if we managed a perfect segmentation, then pass to stop iteration
            elif idx == block_buffer:
                pass  # if all we have left is overlap then it is a perfect segmentation
            else:
                yield tuple(block)[-idx:]  # if we don't have a full block, we must ensure the number of repeats is
                # equal to block buffer
            raise StopIteration


def generate_padded_filename(output_folder: Path, index: int, base: str = "images", digits: int = 2,
                             ext: str = ".tif") -> Path:
    """
    Generates a pathlib Path whose name is defined as '{base}_{index}{ext}' where index is zero-padded if it
    is not equal to the number of digits

    :param output_folder: folder that will contain file
    :type output_folder: pathlib.Path
    :param index: index of file
    :type index: int
    :param base: base tag of file
    :type base: str = "images"
    :param digits: number of digits for representing index
    :type digits: int
    :param ext: file extension
    :type ext: str
    :return: generated filename
    :rtype: pathlib.Path
    """
    index = str(index)

    if len(index) > digits:
        raise ValueError("Index is larger than allocated number of digits in representation")

    while len(index) < digits:
        index = "".join(["0", index])

    return output_folder.joinpath("".join([base, "_", index, ext]))


def wrap_cupy_block(cupy_function: Callable) -> Callable:
    """
    Wraps a cupy function such that incoming numpy arrays are converting to cupy arrays and swapped back on return

    :param cupy_function: any cupy function that accepts numpy arrays
    :type cupy_function: Callable
    :return: wrapped function
    :rtype: Callable
    """
    @wraps(cupy_function)
    def decorator(*args, **kwargs) -> Callable:
        args = list(args)
        if isinstance(args[0], np.ndarray):
            args[0] = cupy.asarray(args[0])
        args = tuple(args)
        return cupy_function(*args, **kwargs).get()
    return decorator


class PatternMatching:
    def __init__(self, value: Any, comparison_expressions: Iterable[Any]):
        """
        Manual implementation of pattern matching for python < 3.10

        Not the most extensible or robust right now, but she works well for the
        current implementations.


        :param value: value or iterator of values of interest
        :type value: Any
        """
        self.value = value
        self.comparison_expressions = comparison_expressions

    def __enter__(self):
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any):
        return False

    def __call__(self, cases: Any):
        """
        Magic call for comparing some case with some value using some comparison operator

        :param cases: case/s for comparison
        :type: Any
        :return: whether the case/s are matched by the value/s
        :rtype: bool
        """
        for value, comparator, case in zip(self.value, self.comparison_expressions, cases):
            if not comparator(value, case):
                return False
        return True
