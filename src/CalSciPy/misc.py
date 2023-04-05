from __future__ import annotations
from typing import Iterable, Iterator, Any
from collections import deque
import numpy as np


def calculate_frames_per_file(y_pixels: int, x_pixels: int, bit_depth: np.dtype = np.uint16, size_cap: float = 3.9) -> int:
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


def generate_padded_filename(output_folder, index: int, base: str = "images",
                             digits: int = 2, ext: str = ".tif")  -> str:
    index = str(index)
    digits = str(digits)

    while len(index) < len(digits):
        index = "".join(["0", index])

    return output_folder.joinpath("".join([base, "_", index, ext]))

class PatternMatching:
    def __init__(self, value: Any, comparison_expressions: Iterable[Any]):
        """
        Manual implementation of pattern matching for python < 3.10

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
        for value, comparator, case in zip(self.value, self.comparison_expressions, cases):
            if not comparator(value, case):
                return False
        return True
