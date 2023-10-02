from __future__ import annotations
from typing import Iterable, Iterator, Callable, Union, Tuple, Sequence
from collections import deque
from numbers import Number
from functools import wraps, partial

from joblib import Parallel, delayed
import numpy as np
from tqdm import tqdm


try:
    import cupy
except ModuleNotFoundError:
    pass


"""
This is where the helpers live that are related to common  mathematical, set, and statistical calculations
"""


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


def min_max_scale(values: Union[Number, Iterable[Number], np.ndarray],
                  old_range: Tuple[Number, Number],
                  new_range: Tuple[Number, Number]
                  ) -> np.ndarray:
    """
    Scale values to new range

    :param values: value/s to be scaled
    :param old_range: old range
    :param new_range: new range
    :return: scaled value
    """
    old_min, old_max = old_range
    new_min, new_max = new_range
    return new_min + ((np.asarray(values) - old_min) * (new_max - new_min)) / (old_max - old_min)


def multiple_random_groups_without_replacement(sample_population: Union[np.ndarray, Sequence],
                                               group_size: int,
                                               num_groups: int
                                               ) -> Tuple:
    """
    Randomly select multiple groups from a population without replacement

    :param sample_population: the population to sample from
    :param group_size: the size of each group drawn
    :param num_groups: the number of groups to draw
    :return: a 1xN tuple containing each group in the order drawn
    """

    sample_population = np.asarray(sample_population)

    selections = []

    for _ in range(num_groups):
        selections.append(np.random.choice(sample_population, size=group_size, replace=False).tolist())
        sample_population = np.setdiff1d(sample_population, np.hstack(selections))

    return tuple([tuple(target) for target in selections])


def sliding_window(sequence: np.ndarray, window_length: int, function: Callable, *args, **kwargs) -> np.ndarray:
    window_gen = generate_sliding_window(range(sequence.shape[-1]), window_length, 1)
    sequence_length = sequence.shape[-1] - window_length + 1
    slider = partial(function, *args, **kwargs)
    values = Parallel(n_jobs=-1, backend="loky", verbose=0)(delayed(slider)(sequence[..., window])
                                                            for window in tqdm(window_gen,
                                                                               total=sequence_length,
                                                                               desc="Calculating sliding windows")
                                                            )
    return np.asarray(values)


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
