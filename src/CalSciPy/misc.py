from __future__ import annotations
from typing import Iterable, Iterator
from collections import deque


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
