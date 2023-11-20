from __future__ import annotations
from typing import Sequence
from pathlib import Path
from itertools import product
from operator import eq

import numpy as np
from PPVD.parsing import find_num_unique_files_containing_tag, find_num_unique_files_given_static_substring
import cv2  # in convoluted determine function

from .._backports import PatternMatching
from .._validators import convert_permitted_types_to_required


def calc_ch_pl_comb(channels: Union[int, Sequence[int]],
                    planes: Union[int, Sequence[int]],
                    exact: bool = True):

    if not isinstance(channels, Sequence):
        if exact:
            channels = [channels, ]
        else:
            channels = range(channels)

    if not isinstance(planes, Sequence):
        if exact:
            planes = [planes, ]
        else:
            planes = range(planes)

    return list(product(channels, planes))


def print_image_description(channels: int,
                            planes: int,
                            frames: int,
                            height: int,
                            width: int
                            ) -> None:
    """
    Function prints the description of an imaging dataset.

    :param channels: number of channels
    :param planes: number of planes
    :param frames: number of frames
    :param height: y-pixels
    :param width: x-pixels
    """
    msg = f"\nImages: {channels * planes * frames}"
    msg += f"\nChannels\t\t{channels}"
    msg += f"\nPlanes\t\t{planes}"
    msg += f"\nFrames\t\t{frames}"
    msg += f"\nHeight\t\t{height}"
    msg += f"\nWidth\t\t{width}"
    msg += "\n"

    print(msg)
