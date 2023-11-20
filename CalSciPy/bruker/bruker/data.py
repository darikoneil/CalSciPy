from __future__ import annotations
from typing import Tuple, Optional, Union, List, Any
from operator import eq
from pathlib import Path
from itertools import product
from xml.etree import ElementTree

import numpy as np
import pandas as pd
import cv2
from PPVD.parsing import convert_permitted_types_to_required, find_num_unique_files_given_static_substring, \
    find_num_unique_files_containing_tag
from PPVD.validation import validate_extension
from tqdm import tqdm as tq

from . import CONSTANTS
from .meta_objects import PhotostimulationMeta
from .factories import BrukerElementFactory
from .._calculations import generate_blocks
from .._files import calculate_frames_per_file#, generate_padded_filename
from ..io_tools import _load_single_tif, _save_single_tif
from .._backports import PatternMatching


DEFAULT_PRAIRIEVIEW_VERSION = CONSTANTS.DEFAULT_PRAIRIEVIEW_VERSION


"""
A collection of functions for importing / converting PrairieView collected data
"""


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
    :returns: a dataframe containing time (index, ms) with aligned columns of voltage recordings/analog data and imaging frame
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


@convert_permitted_types_to_required(permitted=(str, Path), required=Path, pos=0)
def load_bruker_tifs(folder: Union[str, Path],
                     channel: Optional[int] = None,
                     plane: Optional[int] = None
                     ) -> Tuple[np.ndarray]:  # noqa: C901
    """
    This function loads images collected and converted to .tif files by Bruker's Prairieview software.
    If multiple channels or multiple planes exist, each channel and plane combination is loaded to a separate
    numpy array. Identification of multiple channels / planes is dependent on :func:`determine_imaging_content`.
    Images are loaded as unsigned 16-bit (:class:`numpy.uint16`), though note that raw bruker files are
    natively could be 12 or 13-bit.

    :param folder: folder containing a sequence of single frame tiff files
    :param channel: specific channel to load from dataset (zero-indexed)
    :param plane: specific plane to load from dataset (zero-indexed)
    :return: a tuple of numpy arrays (frames, y-pixels, x-pixels, :class:`numpy.uint16`)
     """
    num_channels, num_planes, num_frames, y, x = determine_imaging_content(folder)
    _print_image_description(num_channels, num_planes, num_frames, y, x)

    images = []
    if channel is None and plane is None:
        for channel_, plane_ in product(range(num_channels), range(num_planes)):
            images.append(_load_bruker_tif_stack(folder, channel_, plane_, num_channels, num_planes))
    else:
        images.append(_load_bruker_tif_stack(folder, channel, plane, num_channels, num_planes))
    return tuple(images)


@convert_permitted_types_to_required(permitted=(str, Path), required=str, pos=0)
def load_mark_points(file_path: Union[str, Path], version: str = DEFAULT_PRAIRIEVIEW_VERSION) -> PhotostimulationMeta:
    """

    :param file_path: path to xml file
    :param version: version of prairieview
    :return: photostimulation metadata
    """
    # We generally expected the file to be structured such that
    # File/
    # ├── MarkPointSeriesElements
    # │   └──MarkPointElement
    # │      ├──GalvoPointElement
    # │      |      └──Point
    #        └──GalvoPointElement (Empty)
    # However, by using configurable mappings we do have some wiggle room

    tree = ElementTree.parse(file_path)
    root = tree.getroot()

    # if it's imaging we can grab the version directly
    if "version" in root.attrib:
        version = root.attrib.get("version")

    bruker_element_factory = BrukerElementFactory(version)

    return PhotostimulationMeta(root, factory=bruker_element_factory)


@convert_permitted_types_to_required(permitted=(str, Path), required=Path, pos=0)
def load_voltage_recording(path: Union[str, Path]) -> pd.DataFrame:
    """
    Import bruker analog data from an imaging folder or individual file. By PrairieView naming conventions, these
    files contain "VoltageRecording" in the name.

    :param path: folder or filename containing analog data
    :returns: dataframe containing time (index, ms) x channel data
    """
    if "VoltageRecording" in path.name and path.is_file():
        return _import_csv(path)
    elif not path.is_file():
        file = [file for file in path.glob("*.csv") if "VoltageRecording" in str(file)]
        assert (len(file) == 1), f"Too many files meet parsing requirements: {file}"
        return _import_csv(file)
    else:
        raise FileNotFoundError


@convert_permitted_types_to_required(permitted=(str, Path), required=Path, pos=0)
def load_voltage_output(path: Union[str, Path]) -> pd.DataFrame:
    """
    Import bruker analog data from an imaging folder or individual file. By PrairieView naming conventions, these
    files contain "VoltageOutput" in the name.

    :param path: folder or filename containing analog data
    :returns: dataframe containing time (index, ms) x channel data
    """
    raise NotImplementedError


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
    """
    num_channels, num_planes, num_frames, y, x = determine_imaging_content(input_folder)
    _print_image_description(num_channels, num_planes, num_frames, y, x)

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


@convert_permitted_types_to_required(permitted=(str, Path), required=str, pos=0)
@validate_extension(required_extension=".csv", pos=0)
def _import_csv(path: Union[str, Path]) -> pd.DataFrame:
    """
    Implementation function for loading csv files. Abstracted from the upper-level functions for cleaner logic.

    :param path: filepath of .csv to import
    :return: dataframe containing time (index) x data [0:8]
    """
    data = pd.read_csv(path, skipinitialspace=True)
    data = data.set_index("Time(ms)")
    data.sort_index(inplace=True)
    return data


def _load_bruker_tif_stack(input_folder: Path,
                           channel: int,
                           plane: int,
                           num_channels: int,
                           num_planes: int
                           ) -> np.ndarray:
    """
    Implementation function to load a single bruker tif stack

    :param input_folder: folder containing tif stack
    :param channel: selected channel to load
    :param plane: selected plane to load
    :param num_channels: total number of channels
    :param num_planes: total number of planes
    :returns: loaded images
    """
    naming_convention = generate_bruker_naming_convention(channel, plane, num_channels, num_planes)
    files = list(input_folder.glob(naming_convention))
    pbar = tq(total=len(files))
    pbar.set_description("".join(["Loading Channel ", str(channel), " Plane ", str(plane)]))
    images = [_verbose_load_single_tif(file, pbar) for file in files]
    pbar.close()
    return np.concatenate(images, axis=0)


def _print_image_description(channels: int,
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
    msg = f"\nTotal Images Detected: {channels * planes * frames}"
    msg += f"\nChannels\t\t{channels}"
    msg += f"\nPlanes\t\t{planes}"
    msg += f"\nFrames\t\t{frames}"
    msg += f"\nHeight\t\t{height}"
    msg += f"\nWidth\t\t{width}"
    msg += "\n"

    print(msg)


def _verbose_load_single_tif(file: Union[str, Path], pbar: Any) -> np.ndarray:
    """
    Verbosely loads single tif by running the io_tools load single tif function
    and updating progressbar

    :param file: file to load
    :param pbar: progressbar
    :return: loaded image
    """
    image = _load_single_tif(file)
    pbar.update(1)
    return image
