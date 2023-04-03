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
                block.append(next(iterable))

        except StopIteration:
            # noinspection PyUnboundLocalVariable
            idx += block_buffer
            if idx != block_size:
                yield tuple(block)[-idx:]
            else:
                yield tuple(block)
            raise StopIteration
