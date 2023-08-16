from __future__ import  annotations
from typing import Union
from pathlib import Path
from numbers import Number
from operator import eq

import numpy as np
from PPVD.parsing import convert_permitted_types_to_required

from ._backports import PatternMatching


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


def generate_padded_filename(output_folder: Path,
                             ndex: int, base: str = "images",
                             digits: int = 2,
                             ext: str = ".tif"
                             ) -> Path:
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
# TODO: REDUNDANT WITH ZERO_PAD_NUM_TO_STRING?


def zero_pad_num_to_string(idx: int, length: int) -> str:
    """
    converts an integer index into a zero-padded string with length num_zeros

    :param idx: integer index
    :param length: desired length of the padded string
    :return: a zero-padded string of the index
    """
    str_idx = f"{idx}"

    pad_length = length - len(str_idx)

    return "".join(["_", "0" * pad_length, str_idx])
