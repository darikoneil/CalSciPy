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
from .._files import calculate_frames_per_file #, generate_padded_filename
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
