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

    block = deque(maxlen=block_size)
    iterable = iter(sequence)

    # make first block
    for _ in range(block_size):
        block.append(next(iterable))

    while True:
        try:
            yield tuple(block)

            for _ in range(block_size - block_buffer):
                block.append(next(iterable))

        except StopIteration:
            if block_buffer > 0:
                return tuple(block)[-block_buffer:]
