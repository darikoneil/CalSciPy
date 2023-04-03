from __future__ import annotations
from typing import Tuple, Optional, Union, List
import os
import numpy as np
import pathlib
from collections.abc import Iterable
from prettytable import PrettyTable
from PIL import Image
import cv2
from PPVD.parsing import convert_permitted_types_to_required, find_num_unique_files_given_static_substring, \
    find_num_unique_files_containing_tag
from tqdm.auto import tqdm
from tqdm import tqdm as tq
from tifffile import TiffWriter
import math
from itertools import product


"""
These functions have been incorporated into pyPrairieView and will be deprecated in the future
"""


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


@convert_permitted_types_to_required(permitted=(str, pathlib.Path), required=pathlib.Path, pos=0)
def determine_imaging_content(folder: Union[str, pathlib.Path]) -> Tuple[int, int, int, int, int]:  # noqa: C901
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

    def _find_num_unique_files_containing_tag(tag: str, files: List[pathlib.Path]) -> int:
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


@convert_permitted_types_to_required(permitted=(str, pathlib.Path), required=pathlib.Path, pos=0)
def load_bruker_tiffs(folder: Union[str, pathlib.Path],
                      channels: Optional[int] = None, planes: Optional[int] = None) -> Tuple[np.ndarray]:  # noqa: C901
    """
    Load images collected and exported to .tif by Bruker's Prairieview software to a tuple of numpy arrays.
    If multiple channels or multiple planes exist, each channel and plane combination is loaded to a separate
    numpy array. Identification of multiple channels / planes is dependent on :func:`determine_imaging_content`.
    Images are loaded as unsigned 16-bit, though note that raw bruker files are natively 12 or 13-bit.

    :param folder: folder containing a sequence of single frame tiff files
    :type folder: str or pathlib.Path
    :param channels: specific channel to load from dataset (zero-indexed)
    :type channels: Optional[int] = None
    :param planes: specific plane to load from dataset (zero-indexed)
    :type planes: Optional[int] = None
    :return: All .tif files in the directory loaded to a tuple of numpy arrays
        (frames, y-pixels, x-pixels, :class:`np.uint16`)
    :rtype: tuple[numpy.ndarray]
     """

    def load_images():  # noqa: ANN201
        nonlocal folder
        nonlocal tag
        nonlocal _frames
        nonlocal _y_pixels
        nonlocal _x_pixels
        nonlocal _tqdm_desc

        def load_single_image() -> np.ndarray:
            nonlocal folder
            nonlocal _files
            nonlocal _file
            return cv2.imread(str(_files[_file]), flags=-1)

        tag = "".join(["*", tag, "*"])
        _files = [_file for _file in folder.glob("*.tif") if _file.match(tag)]
        if len(_files) == 0:
            return
        # Bruker is usually saved to .tif in 0-65536 (uint16) even though recording says 8192 (uint13)
        complete_image = np.full((_frames, _y_pixels, _x_pixels), 0, dtype=np.uint16)
        for _file in tqdm(
                range(_frames),
                total=_frames,
                desc=_tqdm_desc,
                disable=False,
        ):
            complete_image[_file, :, :] = load_single_image()

        return complete_image

    _channels, _planes, _frames, _y_pixels, _x_pixels = determine_imaging_content(folder)
    _pretty_print_image_description(_channels, _planes, _frames, _y_pixels, _x_pixels)

    if channels is None:
        channels = range(2)
    if planes is None:
        planes = range(_planes)

    images = []  # append all channel/plane combos to this list

    # if planes is 1 bruker uses "ChX_XXXXXX" designator channel+frames
    if _planes == 1:
        if not isinstance(channels, Iterable):
            _tqdm_desc = "".join(["Loading Plane ", str(planes),
                                  " Channel ", str(channels), " images..."])
            tag = "".join(["Ch", str(channels + 1)])
            images.append(load_images())
            images = filter(lambda image: image is not None, images)
            return tuple(images)
        for _channel, _plane in product(channels, planes):
            _tqdm_desc = "".join(["Loading Plane ", str(_plane),
                                  " Channel ", str(_channel), " images..."])
            tag = "".join(["Ch", str(_channel + 1)])
            images.append(load_images())
        images = filter(lambda image: image is not None, images)
        return tuple(images)

    if not isinstance(planes, Iterable):
        if not isinstance(channels, Iterable):
            _tqdm_desc = "".join(["Loading Plane ", str(planes),
                                  " Channel ", str(channels), " images..."])
            tag = "".join(["Ch", str(channels + 1), "_00000", str(planes + 1)])
            images.append(load_images())
            images = filter(lambda image: image is not None, images)
            return tuple(images)
        for _channel in channels:
            _tqdm_desc = "".join(["Loading Plane ", str(planes),
                                  " Channel ", str(_channel), " images..."])
            tag = "".join(["Ch", str(_channel + 1), "_00000", str(planes + 1)])
            images.append(load_images())
        images = filter(lambda image: image is not None, images)
        return tuple(images)

    # if planes > 1 bruker uses "CycleXXXXX_ChX_XXXXXX" designator for frames+channel+plane
    for _channel, _plane in product(channels, planes):
        _tqdm_desc = "".join(["Loading Plane ", str(_plane),
                              " Channel ", str(_channel), " images..."])
        tag = "".join(["Ch", str(_channel + 1), "_00000", str(_plane + 1)])
        images.append(load_images())
    images = filter(lambda image: image is not None, images)
    return tuple(images)
# TODO REFACTOR THIS SPAGHETTI


def repackage_bruker_tiffs(input_folder: Union[str, pathlib.Path], output_folder: Union[str, pathlib.Path],
                           *args: Union[int, tuple[int]]) -> None:  # noqa: C901
    """
    Repackages a folder containing .tif files exported by Bruker's Prairieview software into a sequence of <4 GB .tif
    stacks.

    :param input_folder: folder containing a sequence of single frame .tif files exported by Bruker's Prairieview
    :type input_folder: str or pathlib.Path
    :param output_folder: empty folder where .tif stacks will be saved
    :type output_folder: str or pathlib.Path
    :param args: optional argument to indicate the repackaging of a specific channel and/or plane
    :type args: int
    :rtype: None
    """

    # code here is pretty rough, needs TLC. ~horror~
    # noqa: C901
    def load_image() -> np.ndarray:
        nonlocal input_folder
        nonlocal _files
        nonlocal _file
        nonlocal _offset
        return cv2.imread(str(_files[_file + _offset]), flags=-1)

    def save_image(images_: np.ndarray, path_: str, type_: np.dtype = np.uint16) -> None:
        if len(images_.shape) == 2:
            with TiffWriter(path_) as tif:
                tif.write(np.floor(images_).astype(type_))
            return

        with TiffWriter(path_) as tif:
            for frame in np.floor(images_).astype(type_):
                tif.write(frame)

    def find_files(tag: Union[str, list[str]]) -> list:
        nonlocal input_folder
        nonlocal _files

        def check_file_contents(tag_: str, file_: pathlib.WindowsPath) -> bool:
            if tag_ in str(file_.stem).split("_"):
                return True
            else:
                return False

        # reset, rather maintain code as is then make a new temporary variable since this
        # is basically instant
        _files = [_file for _file in pathlib.Path(input_folder).glob("*.tif")]  # noqa: C416

        # now find
        if isinstance(tag, list):
            tag = "".join([tag[0], "_", tag[1]])
            _files = [_file for _file in _files if tag in str(_file.stem)]
        else:
            _files = [_file for _file in _files if check_file_contents(tag, _file)]  # noqa: C416

    def not_over_4gb() -> bool:
        nonlocal _files
        nonlocal _y
        nonlocal _x

        gb = np.full((_files.__len__(), _y, _x), 1, dtype=np.uint16).nbytes
        if gb <= 3.9:  # 3.9 as a safety buffer
            return True
        else:
            return False

    _files = [_file for _file in pathlib.Path(input_folder).rglob("*.tif")]  # noqa: C416
    _channels, _planes, _frames, _y, _x = determine_imaging_content(input_folder)
    _pretty_print_image_description(_channels, _planes, _frames, _y, _x)

    # finding the files for a specific channel/plane here
    if args:
        # unpack if necessary
        if isinstance(args[0], tuple):
            _c = args[0][0]
            _p = args[0][1]
            args = (_c, _p)

        if _channels > 1 and _planes > 1 and len(args) >= 2:
            _base_tag = "00000"
            _tag = ["".join(["Ch", str(args[0] + 1)]), "".join([_base_tag, str(args[1] + 1), ".ome"])]
            find_files(_tag)
        elif _channels == 1 and _planes > 1 and len(args) == 1:
            _base_tag = "00000"
            _tag = "".join([_base_tag, str(args[0] + 1), ".ome"])
            find_files(_tag)
            print(_files)
        elif _channels == 1 and _planes > 1 and len(args) >= 2:
            _base_tag = "00000"
            _tag = "".join([_base_tag, str(args[1] + 1), ".ome"])
            find_files(_tag)
        elif _channels > 1 and _planes == 1:
            _tag = "".join(["Ch", str(args[0] + 1)])
            find_files(_tag)
        else:
            pass
    else:
        if not _channels == 1:
            raise AssertionError("Folder contains multiple channels")
        if not _planes == 1:
            raise AssertionError("Folder contains multiple planes")

    # Decide whether saving in single stack is possible
    if not_over_4gb():
        _images = np.full((_frames, _y, _x), 0, dtype=np.uint16)
        _offset = 0
        for _file in range(_frames):
            _images[_file, :, :] = load_image()
        save_image(_images, os.path.join(output_folder, "compiled_video_0_of_0.tif"))
        return
    else:
        # noinspection PyTypeChecker
        _chunks = math.ceil(_frames / 7000)
        c_idx = 1
        _offset = int()

        _pbar = tq(total=_frames)
        _pbar.set_description("Repackaging Bruker Tiffs...")

        for _chunk in range(0, _frames, 7000):

            _start_idx = _chunk
            _offset = _start_idx
            _end_idx = _chunk + 7000
            _chunk_frames = _end_idx - _start_idx
            # If this is the last chunk which may not contain a full 7000 frames...
            if _end_idx > _frames:
                _end_idx = _frames
                _chunk_frames = _end_idx - _start_idx
                _end_idx += 1

            image_chunk = np.full((_chunk_frames, _y, _x), 0, dtype=np.uint16)

            for _file in range(_chunk_frames):
                image_chunk[_file, :, :] = load_image()
                _pbar.update(1)

            if c_idx < 10:
                save_image(image_chunk, os.path.join(output_folder, "".join([
                    "compiled_video_0", str(c_idx), "_of_", str(_chunks), ".tif"])))
            else:
                save_image(image_chunk, os.path.join(output_folder, "".join([
                    "compiled_video_", str(c_idx), "_of_", str(_chunks), ".tif"])))
            c_idx += 1
        _pbar.close()
    return
# TODO REFACTOR THIS SPAGHETTI
