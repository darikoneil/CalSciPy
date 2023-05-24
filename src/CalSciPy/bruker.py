from __future__ import annotations
from typing import Tuple, Optional, Union, List, Any
import numpy as np
from pathlib import Path
from prettytable import PrettyTable
import cv2
from PPVD.parsing import convert_permitted_types_to_required, find_num_unique_files_given_static_substring, \
    find_num_unique_files_containing_tag
from tqdm import tqdm as tq
from itertools import product
from .misc import PatternMatching, calculate_frames_per_file, generate_blocks, generate_padded_filename
from operator import eq
from .io_tools import _load_single_tif, _save_single_tif


"""
These functions have been incorporated into pyPrairieView and will be deprecated in the future
"""


def generate_bruker_naming_convention(channel: int = 0, plane: int = 0, num_channels: int = 1, num_planes: int = 1) \
        -> str:
    """
    Generates the expected bruker naming convention for images collected with an arbitrary number of cycles & channels

    This function expects that the naming convention is {experiment_name}_Cycle00000_Ch0_000000.ome.tiff
    where the channel is one-indexed. The 5-digit cycle id represents the frame if using multiplane imaging and
    the 6-digit tag represents the plane. Otherwise, the 5-digit tag is static and the 6-digit tag represents the frame.
    Channel and plane are *zero-indexed*.

    :param channel: channel to produce name for
    :type channel: int = 0
    :param plane: plane to produce name for
    :type plane: int = 0
    :param num_channels: number of channels
    :type num_channels: int = 1
    :param num_planes: number of planes
    :type num_planes: int = 1
    :return:
    """
    # if channel + 1 > num_channels:
    #    raise AssertionError("Channel selection exceeds the amount of identified channels")
    # if plane + 1 > num_planes:
    #    raise AssertionError("Plane selection exceeds the amount of identified planes")
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


@convert_permitted_types_to_required(permitted=(str, Path), required=Path, pos=0)
def determine_imaging_content(folder: Union[str, Path]) -> Tuple[int, int, int, int, int]:  # noqa: C901
    """
    This function determines the number of channels and planes within a folder containing .tif files
    exported by Bruker's Prairieview software. It also determines the size of the images (frames, y-pixels, x-pixels).
    It's a quick / fast alternative to parsing its respective xml. Dependent on the naming conventions of PrairieView.

    :param folder: folder containing bruker imaging data
    :type folder: str or pathlib.Path
    :returns: channels, planes, frames, height, width
    :rtype: tuple[int, int, int, int, int]
    """

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


@convert_permitted_types_to_required(permitted=(str, Path), required=Path, pos=0)
def load_bruker_tifs(folder: Union[str, Path],
                     channel: Optional[int] = None, plane: Optional[int] = None) -> Tuple[np.ndarray]:  # noqa: C901
    """
    Load images collected and exported to .tif by Bruker's Prairieview software to a tuple of numpy arrays.
    If multiple channels or multiple planes exist, each channel and plane combination is loaded to a separate
    numpy array. Identification of multiple channels / planes is dependent on :func:`determine_imaging_content`.
    Images are loaded as unsigned 16-bit, though note that raw bruker files are natively 12 or 13-bit.

    :param folder: folder containing a sequence of single frame tiff files
    :type folder: str or pathlib.Path
    :param channel: specific channel to load from dataset (zero-indexed)
    :type channel: Optional[int] = None
    :param plane: specific plane to load from dataset (zero-indexed)
    :type plane: Optional[int] = None
    :return: All .tif files in the directory loaded to a tuple of numpy arrays
        (frames, y-pixels, x-pixels, :class:`np.uint16`)
    :rtype: tuple[numpy.ndarray]
     """
    num_channels, num_planes, num_frames, y, x = determine_imaging_content(folder)
    _pretty_print_image_description(num_channels, num_planes, num_frames, y, x)

    images = []
    if channel is None and plane is None:
        for channel_, plane_ in product(range(num_channels), range(num_planes)):
            images.append(_load_bruker_tif_stack(folder, channel_, plane_, num_channels, num_planes))
    else:
        images.append(_load_bruker_tif_stack(folder, channel, plane, num_channels, num_planes))
    return tuple(images)


@convert_permitted_types_to_required(permitted=(str, Path), required=Path, pos=0)
def repackage_bruker_tifs(input_folder: Union[str, Path], output_folder: Union[str, Path],
                          channel: int = 0, plane: int = 0) -> None:
    """
    Repackages a folder containing .tif files exported by Bruker's Prairieview software into a sequence of <4 GB .tif
    stacks. Channels are planes are **zero-indexed**.

    :param input_folder: folder containing a sequence of single frame .tif files exported by Bruker's Prairieview
    :type input_folder: str or pathlib.Path
    :param output_folder: empty folder where .tif stacks will be saved
    :type output_folder: str or pathlib.Path
    :param channel: optional input specifying channel
    :type channel: int = 0
    :param plane: optional input specifying plane
    :type plane: int = 0
    :rtype: None
    """
    num_channels, num_planes, num_frames, y, x = determine_imaging_content(input_folder)
    _pretty_print_image_description(num_channels, num_planes, num_frames, y, x)

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

            filename = generate_padded_filename(output_folder, stack_index)
            _save_single_tif(filename, np.concatenate(images, axis=0))

            stack_index += 1
    else:
        images = [_verbose_load_single_tif(file, pbar) for file in files]
        images = np.concatenate(images, axis=0)
        _save_single_tif(output_folder.joinpath("images.tif"), images)

    pbar.close()


def _load_bruker_tif_stack(input_folder: Path, channel: int, plane: int,
                           num_channels: int, num_planes: int) -> np.ndarray:
    """
    Implementation function to load a single bruker tif stack

    :param input_folder: folder containing tif stack
    :type input_folder: pathlib.Path
    :param channel: selected channel to load
    :type channel: int
    :param plane: selected plane to load
    :type plane: int
    :param num_channels: total number of channels
    :type num_channels: int
    :param num_planes: total number of planes
    :type num_planes: int
    :return: loaded images
    :rtype: numpy.ndarray
    """
    naming_convention = generate_bruker_naming_convention(channel, plane, num_channels, num_planes)
    files = list(input_folder.glob(naming_convention))
    pbar = tq(total=len(files))
    pbar.set_description("".join(["Loading Channel ", str(channel), " Plane ", str(plane)]))
    images = [_verbose_load_single_tif(file, pbar) for file in files]
    pbar.close()
    return np.concatenate(images, axis=0)


def _pretty_print_image_description(channels: int, planes: int, frames: int, height: int, width: int) -> None:
    """
    Function prints the description of an imaging dataset as a table.

    :param channels: number of channels
    :type channels: int
    :param planes: number of planes
    :type planes: int
    :param frames: number of frames
    :type frames: int
    :param height: y-pixels
    :type height: int
    :param width: x-pixels
    :type width: int
    :rtype: None
    """
    _table = PrettyTable()
    _table.header = False
    _table.add_row(["Total Images Detected", channels * planes * frames])
    _table.add_row(["Channels", channels])
    _table.add_row(["Planes", planes])
    _table.add_row(["Frames", frames])
    _table.add_row(["Height", height])
    _table.add_row(["Width", width])
    print("\n")
    print(_table)


def _verbose_load_single_tif(file: Union[str, Path], pbar: Any) -> np.ndarray:
    """
    Verbosely loads single tif by running the io_tools load single tif function
    and updating progressbar

    :param file: file to load
    :type file: str or pathlib.Path
    :param pbar: progressbar
    :type pbar: Any
    :return: loaded image
    :rtype: numpy.ndarray
    """
    image = _load_single_tif(file)
    pbar.update(1)
    return image
