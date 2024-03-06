from __future__ import annotations
from typing import Union, Tuple, List
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pandas as pd
from tqdm import tqdm as tq
import cv2  # in convoluted determine function

from ..io_tools import _save_single_tif, _verbose_load_single_tif
from .._calculations import generate_blocks
from .._files import (calculate_frames_per_file, find_num_unique_files_containing_tag,
                      find_num_unique_files_given_static_substring, zero_pad_num_to_string)
from .._validators import convert_permitted_types_to_required
from ._helpers import generate_bruker_naming_convention, print_image_description


def align_data(analog_data: pd.DataFrame,
               frame_times: pd.DataFrame,
               fill: bool = False,
               method: str = "nearest"
               ) -> pd.DataFrame:
    """
    Synchronizes analog data & imaging frames using the timestamp of each frame. Option to generate a second column
    in which the frame index is interpolated such that each analog sample matches with an associated frame.

    :param analog_data: analog data
    :param frame_times: frame timestamps
    :param fill: whether to include an interpolated column so each sample has an associated frame
    :param method: method for interpolating samples
    :returns: a dataframe containing time (index, ms) with aligned columns of voltage recordings/analog data and
        imaging frame

    .. versionadded:: 0.8.1

    .. warning:: Currently untested

    """
    frame_times = frame_times.reindex(index=analog_data.index)

    # Join frames & analog (deep copy to make sure not a view)
    data = analog_data.copy(deep=True)
    data = data.join(frame_times)

    if fill:
        frame_times_filled = frame_times.copy(deep=True)
        frame_times_filled.columns = ["Imaging Frame (interpolated)"]
        frame_times_filled.interpolate(method=method, inplace=True)
        # forward fill the final frame
        frame_times_filled.ffill(inplace=True)
        data = data.join(frame_times_filled)

    return data


@convert_permitted_types_to_required(permitted=(str, Path), required=Path, pos=0)
def determine_imaging_content(folder: Union[str, Path]) -> Tuple[int, int, int, int, int]:  # noqa: C901
    """
    This function determines the number of channels and planes within a folder containing .tif files
    exported by Bruker's Prairieview software. It also determines the size of the images (frames, y-pixels, x-pixels).
    It's a quick / fast alternative to parsing its respective xml. However, note that the function is dependent on the
    naming conventions of PrairieView and will not work on arbitrary folders.

    :param folder: folder containing bruker imaging data
    :returns: channels, planes, frames, height, width

    .. versionadded:: 0.8.1

    .. warning:: Currently untested

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


@convert_permitted_types_to_required(permitted=(str, Path), required=str, pos=0)
def extract_frame_times(filename: Union[str, Path]) -> pd.DataFrame:
    """
    Function to extract the relative frame times from a PrairieView imaging session's primary .xml file

    :param: filename
    :returns: dataframe containing time (index, ms) x imaging frame (*zero-indexed*)

    .. versionadded:: 0.8.1

    .. warning:: Currently untested

    """
    tree = ElementTree.parse(filename)
    root = tree.getroot()

    # assert expected
    # child_tags = [child.tag for child in root]
    # expected_tags = ("SystemIDs", "PVStateShard", "Sequence")

    # for tag_ in expected_tags:
    #   assert (tag_ in child_tags), "XML follows unexpected structure"

    # Since expected, let's grab frame sequence
    sequence = root.find("Sequence")
    # use set comprehension to avoid duplicates
    relative_frame_times = {frame.attrib.get("relativeTime") for frame in sequence if "relativeTime" in frame.attrib}
    # convert to float (appropriate type) & sort chronologically
    relative_frame_times = sorted([float(frame) for frame in relative_frame_times])
    frames = range(len(relative_frame_times))
    # convert to same type as analog data to avoid people getting gotcha'd by pandas
    frames = np.array(frames).astype(np.float64)
    # convert to milliseconds, create new array to avoid people getting gotcha'd by pandas
    frame_times = np.array(relative_frame_times) * 1000
    # round to each millisecond
    frame_times = np.round(frame_times).astype(np.int64)
    # make index
    frame_times = pd.Index(data=frame_times, name="Time (ms)")
    # make dataframe
    return pd.DataFrame(data=frames, index=frame_times, columns=["Imaging Frame"])


@convert_permitted_types_to_required(permitted=(str, Path), required=Path, pos=0)
def repackage_bruker_tifs(input_folder: Union[str, Path],
                          output_folder: Union[str, Path],
                          channel: int = 0,
                          plane: int = 0
                          ) -> None:
    """
    This function repackages a folder containing .tif files exported by Bruker's Prairieview software into a sequence
    of <4 GB .tif stacks. Note that parameters channel and plane are **zero-indexed**.

    :param input_folder: folder containing a sequence of single frame .tif files exported by Bruker's Prairieview
    :param output_folder: empty folder where .tif stacks will be saved
    :param channel: specify channel
    :param plane: specify plane

    .. versionadded:: 0.8.1

    .. warning:: Currently untested

    """
    num_channels, num_planes, num_frames, y, x = determine_imaging_content(input_folder)
    print_image_description(num_channels, num_planes, num_frames, y, x)

    naming_convention = generate_bruker_naming_convention(channel, plane, num_channels, num_planes)

    files = list(input_folder.glob(naming_convention))

    pbar = tq(total=num_frames)
    pbar.set_description("Repackaging...")

    block_size = calculate_frames_per_file(y, x)

    if num_frames > block_size:

        block_buffer = 0
        frame_index = list(range(num_frames))
        blocks = generate_blocks(frame_index, block_size, block_buffer)
        stack_index = 0

        for block in blocks:
            images = []
            for frame in block:
                images.append(_verbose_load_single_tif(files[frame], pbar))

            filename = output_folder.joinpath("".join(["images_",
                                                       zero_pad_num_to_string(stack_index, 2),
                                                       ".tif"]))
            _save_single_tif(filename, np.concatenate(images, axis=0))

            stack_index += 1
    else:
        images = [_verbose_load_single_tif(file, pbar) for file in files]
        images = np.concatenate(images, axis=0)
        _save_single_tif(output_folder.joinpath("images.tif"), images)

    pbar.close()
