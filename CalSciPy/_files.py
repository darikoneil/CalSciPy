from __future__ import annotations
from typing import Union, List
from pathlib import Path
from numbers import Number
from operator import eq

import numpy as np

from ._backports import PatternMatching
from ._validators import convert_permitted_types_to_required


"""
This is where the helpers for manipulating filenames and other file-oriented utilities live
"""


def calculate_frames_per_file(y_pixels: int,
                              x_pixels: int,
                              bit_depth: np.dtype = np.uint16,
                              size_cap: Number = 3.9
                              ) -> int:
    """
    Estimates the number of image frames to allocate to each file given some maximum size.

    :param y_pixels: number of y_pixels in image
    :param x_pixels: number of x_pixels in image
    :param bit_depth: bit-depth / type of image elements
    :param size_cap: maximum file size (in GB)
    :returns: the maximum number of frames to allocate for each file
    """
    single_frame_size = np.ones((y_pixels, x_pixels), dtype=bit_depth).nbytes * 1e-9
    return size_cap // single_frame_size


@convert_permitted_types_to_required(permitted=(str, Path), required=Path, pos=0)
def check_filepath(path: Union[str, Path], name: str = None, extension: str = None, default_name: str = None) -> Path:
    """
    Process provide filepath and ensure appropriate for saving (e.g., create folder)

    :param path: path provided
    :param name: name provided
    :param extension: desired extension
    :param default_name: default name
    :return: ready-to-use filename
    """
    if name is None:
        name = default_name

    with PatternMatching(
            [path.suffix, path.exists()],
            [eq, eq]
    ) as case:
        # if file and exists
        if case([extension, True]):
            return path.with_suffix(extension)
        # if file and does not exist
        elif case([extension, False]):
            return path.with_suffix(extension)
        # if folder and exists:
        elif case([False, True]):
            return path.joinpath(name).with_suffix(extension)
        # if folder and does not exist
        else:
            path.mkdir(parents=True, exist_ok=True)
            return path.joinpath(name).with_suffix(extension)


def check_split_strings(tag_: str, str_to_split: str) -> str:
    return [_tag for _tag in str_to_split.split("_") if tag_ in _tag]
# REFACTOR


def find_num_unique_files_given_static_substring(tag: str, files: List[Path]) -> int:
    _hits = [check_split_strings(tag, str(_file)) for _file in files]
    _hits = [_hit for _nested_hit in _hits for _hit in _nested_hit]
    return list(_hits).__len__()
# REFACTOR


def find_num_unique_files_containing_tag(tag: str, files: List[Path]) -> int:
    _hits = [check_split_strings(tag, str(_file)) for _file in files]
    _hits = [_hit for _nested_hit in _hits for _hit in _nested_hit]
    return list(dict.fromkeys(_hits)).__len__()


# REFACTOR
def zero_pad_num_to_string(idx: int, length: int) -> str:
    """
    converts an integer index into a zero-padded string with length num_zeros

    :param idx: integer index
    :param length: desired length of the padded string
    :returns: a zero-padded string of the index
    """
    str_idx = f"{idx}"

    pad_length = length - len(str_idx)

    if pad_length < 0:
        raise ValueError("Index is larger than allocated number of digits in representation")

    return "".join(["_", "0" * pad_length, str_idx])
