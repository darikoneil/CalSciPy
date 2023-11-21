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


@convert_permitted_types_to_required(permitted=(str, Path), required=Path, pos=0)
def determine_imaging_content(folder: Union[str, Path]) -> Tuple[int, int, int, int, int]:  # noqa: C901
    """
    This function determines the number of channels and planes within a folder containing .tif files
    exported by Bruker's Prairieview software. It also determines the size of the images (frames, y-pixels, x-pixels).
    It's a quick / fast alternative to parsing its respective xml. However, note that the function is dependent on the
    naming conventions of PrairieView and will not work on arbitrary folders.

    :param folder: folder containing bruker imaging data
    :returns: channels, planes, frames, height, width
    """

    # TODO: I am spaghetti

    _files = [_file for _file in folder.glob("*.tif") if _file.is_file()]

    def _extract_file_identifiers(str_to_split: str) -> list:
        return str_to_split.split("_")[-3:]

    def _check_file_identifiers(tag_: str, str_to_split: str) -> str:
        return [_tag for _tag in _extract_file_identifiers(str_to_split) if tag_ in _tag]

    def _find_num_unique_files_containing_tag(tag: str, files: List[Path]) -> int:
        _hits = [_check_file_identifiers(tag, str(_file)) for _file in files]
        _hits = [_hit for _nested_hit in _hits for _hit in _nested_hit]
        return list(dict.fromkeys(_hits)).__len__()

    def find_channels() -> int:
        nonlocal _files
        return _find_num_unique_files_containing_tag("Ch", _files)

    def is_multiplane() -> bool:
        nonlocal _files
        if find_num_unique_files_containing_tag("Cycle0000", _files) > 1:
            return True
        else:
            return False

    def find_planes() -> int:
        nonlocal channels
        nonlocal _files
        if is_multiplane():
            return find_num_unique_files_given_static_substring("Cycle00001", _files) // channels
        else:
            return 1

    def find_frames() -> int:
        nonlocal channels
        if is_multiplane():
            return find_num_unique_files_given_static_substring("000001.ome", _files) // channels
        else:
            return find_num_unique_files_given_static_substring("Cycle00001", _files) // channels

    def find_dimensions() -> Tuple[int, int]:
        nonlocal folder
        nonlocal _files
        return cv2.imread("".join(str(_files[0])), flags=-1).shape

    channels = find_channels()

    # noinspection PyTypeChecker
    return channels, find_planes(), find_frames(), find_dimensions()[0], find_dimensions()[1]
    # apparently * on a return is illegal for python 3.7  # noqa


def generate_bruker_naming_convention(channel: int,
                                      plane: int,
                                      num_channels: int = 1,
                                      num_planes: int = 1
                                      ) -> str:
    """
    Generates the expected bruker naming convention for images collected with an arbitrary number of cycles & channels

    This function expects that the naming convention is _Cycle00000_Ch0_000000.ome.tiff where the channel is
    one-indexed. The 5-digit cycle id represents the frame if using multiplane imaging and the 6-digit tag represents
    the plane. Otherwise, the 5-digit tag is static and the 6-digit tag represents the frame.

    Please note that the parameters channel and plane are *zero-indexed*.

    :param channel: channel to produce name for
    :param plane: plane to produce name for
    :param num_channels: number of channels
    :param num_planes: number of planes
    :returns: proper naming convention
    """
    if num_channels > 1:
        num_channels = 2
    if num_planes > 1:
        num_planes = 2
    with PatternMatching([num_channels, num_planes], [eq, eq]) as case:
        if case([1, 1]):
            return "*.ome.tif"
        elif case([2, 1]):
            return "".join(["*Ch", str(channel + 1), "*"])
        elif case([1, 2]):
            return "".join(["*00000", str(plane + 1), ".ome.tif"])
        elif case([2, 2]):
            return "".join(["*Ch", str(channel + 1), "00000", str(plane + 1), ".ome.tif"])
