from __future__ import annotations
from typing import Iterable, Iterator, Any, Callable
from collections import deque
from pathlib import Path
from numbers import Number
from functools import wraps, partial
from joblib import Parallel, delayed
import numpy as np
from tqdm import tqdm
from numbers import Number


try:
    import cupy
except ModuleNotFoundError:
    pass


def generate_time_vector(num_samples: int, sampling_frequency: Number = 30.0, start: Number = 0.0, step: Number = None,
) -> np.ndarray:
    """
    Generates a time vector for a number of samples collected at either

    :param num_samples:
    :param sampling_frequency:
    :param start:
    :param step:
    :return:
    """

    if not step:
        step = 1 /sampling_frequency

    return np.arange(0, num_samples) * step + start


def calculate_frames_per_file(y_pixels: int, x_pixels: int, bit_depth: np.dtype = np.uint16, size_cap: Number = 3.9) \
        -> int:
    """
    Estimates the number of image frames to allocate to each file given some maximum size.

    :param y_pixels: number of y_pixels in image
    :param x_pixels: number of x_pixels in image
    :param bit_depth: bit-depth / type of image elements
    :param size_cap: maximum file size
    :returns: the maximum number of frames to allocate for each file
    """
    single_frame_size = np.ones((y_pixels, x_pixels), dtype=bit_depth).nbytes * 1e-9
    return size_cap // single_frame_size


def generate_blocks(sequence: Iterable, block_size: int, block_buffer: int = 0) -> Iterator:
    """
    Returns a generator of some arbitrary iterable sequence that yields m blocks with overlapping regions of size n

    :param sequence: Sequence to be split into overlapping blocks
    :param block_size: size of blocks
    :param block_buffer: size of overlap between blocks
    :returns: generator yielding m blocks with overlapping regions of size n
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


def generate_overlapping_blocks(sequence: Iterable, block_size: int, block_buffer: int) -> Iterator:
    """
    Returns a generator of some arbitrary iterable sequence that yields m blocks with overlapping regions of size n

    :param sequence: Sequence to be split into overlapping blocks
    :param block_size: size of blocks
    :param block_buffer: size of overlap between blocks
    :returns: generator yielding m blocks with overlapping regions of size n
    """
    if block_size <= 1:
        raise ValueError("Block size must be > 1")
    if block_buffer >= block_size:
        raise AssertionError("Block buffer must be smaller than the size of the block.")
    if block_size >= len(sequence):
        raise AssertionError("Block must be smaller than iterable sequence")

    block_size = int(block_size)  # coerce in case float

    block = deque(maxlen=block_size + 2 * block_buffer)
    iterable = iter(sequence)

    # make nil block
    for _ in range(block_size + 2 * block_buffer):
        block.append(0)

    # make first block
    for _ in range(block_size + block_buffer):
        block.append(next(iterable))

    while True:
        try:
            yield tuple(block)
            for idx in range(block_size):  # noqa: B007
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
    :param index: index of file
    :param base: base tag of file
    :param digits: number of digits for representing index
    :param ext: file extension
    :returns: generated filename
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
    :return: wrapped function
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
        :returns: whether the case/s are matched by the value/s
        """
        for value, comparator, case in zip(self.value, self.comparison_expressions, cases):
            if not comparator(value, case):
                return False
        return True


def sliding_window(sequence: np.ndarray, window_length: int, function: Callable, *args, **kwargs) -> np.ndarray:
    window_gen = generate_sliding_window(range(sequence.shape[-1]), window_length, 1)
    sequence_length = sequence.shape[-1] - window_length + 1
    slider = partial(function, *args, **kwargs)
    values = Parallel(n_jobs=-1, backend="loky", verbose=0)\
        (delayed(slider)(sequence[..., window])
         for window in tqdm(window_gen, total=sequence_length, desc="Calculating sliding windows"))
    return np.asarray(values)


def generate_sliding_window(sequence: Iterable, window_length: int, step_size: int = 1) -> np.ndarray:

    window = deque(maxlen=window_length)
    iterable = iter(sequence)

    for _ in range(window_length):
        window.append(next(iterable))

    while True:
        try:
            yield tuple(window)
            for idx in range(step_size):  # noqa: B007
                window.append(next(iterable))

        except StopIteration:
            # noinspection PyUnboundLocalVariable
            if idx == 0:
                pass
            elif idx == step_size:
                pass
            else:
                return tuple(window)
            return
