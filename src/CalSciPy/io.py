from __future__ import annotations
import itertools
import numpy as np
import os
from PIL import Image
from tqdm.auto import tqdm
from tqdm import tqdm as tq
import tifffile  # TODO CAN PROBABLY PHASE OUT AS REQUIREMENT
from typing import Tuple, Optional, Union
import math
import pathlib
from prettytable import PrettyTable
from imageio import mimwrite
from collections.abc import Iterable

from ._validation import validate_exists, validate_extension, validate_filename, validate_path
from ._parsing import convert_optionals, if_dir_append_filename, if_dir_join_filename, require_full_path


@convert_optionals(permitted=(str, pathlib.Path), required=pathlib.Path)
@validate_path(pos=0)
@validate_exists(pos=0)
def determine_bruker_folder_contents(folder: Union[str, pathlib.Path]) -> Tuple[int, int, int, int, int]:
    """
    Function determine contents of the bruker folder

    :param folder: Folder containing bruker imaging data
    :type folder: Union[str, pathlib.Path]
    :returns: channels, planes, frames, Height, Width
    :rtype: tuple
    """

    _files = [_file for _file in folder.glob("*.tif") if _file.is_file()]

    def parser(tag_: str, split_strs: list) -> str:
        return [_tag for _tag in split_strs if tag_ in _tag]

    def find_num_unique_strings_given_static_substring(tag: str) -> int:
        nonlocal _files
        _hits = [parser(tag, str(_file).split("_")) for _file in _files]
        _hits = [_hit for _nested_hit in _hits for _hit in _nested_hit]
        return list(_hits).__len__()

    def find_num_unique_substring_containing_tag(tag: str) -> int:
        nonlocal _files
        _hits = [parser(tag, str(_file).split("_")) for _file in _files]
        _hits = [_hit for _nested_hit in _hits for _hit in _nested_hit]
        return list(dict.fromkeys(_hits)).__len__()

    def find_channels() -> int:
        return find_num_unique_substring_containing_tag("Ch")

    def is_multiplane() -> bool:
        if find_num_unique_substring_containing_tag("Cycle0000") > 1:
            return True
        else:
            return False

    def find_planes() -> int:
        nonlocal channels

        if is_multiplane():
            return find_num_unique_strings_given_static_substring("Cycle00001")//channels
        else:
            return 1

    def find_frames() -> int:
        nonlocal channels
        if is_multiplane():
            return find_num_unique_strings_given_static_substring("000001.ome")//channels
        else:
            return find_num_unique_strings_given_static_substring("Cycle00001")//channels

    def find_dimensions() -> Tuple[int, int]:
        nonlocal folder
        nonlocal _files
        return np.asarray(Image.open("".join(str(_files[0])))).shape

    channels = find_channels()

    # noinspection PyTypeChecker
    return (channels, find_planes(), find_frames(), *find_dimensions())


@convert_optionals(permitted=(str, pathlib.Path), required=str, pos=0)
@validate_path(pos=0)
@validate_exists(pos=0)
def load_all_tiffs(folder: Union[str, pathlib.Path]) -> np.ndarray:
    """
    Load a sequence of tiff stacks

    :param folder: Folder containing a sequence of tiff stacks
    :type folder: Union[str, pathlib.Path]
    :return: complete_image numpy array [Z x Y x X] as uint16
    :rtype: Any
    """
    if isinstance(folder, pathlib.Path):
        folder = str(folder)

    _filenames = [str(_filename.name) for _filename in pathlib.Path(folder).glob("*") if ".tif" in _filename.suffix]
    y_pix, x_pix = tifffile.TiffFile(folder + "\\" + _filenames[0]).pages[0].shape
    _num_frames = []  # initialize
    [_num_frames.append(len(tifffile.TiffFile(folder + "\\" + _filename).pages)) for _filename in _filenames]
    _total_frames = sum(_num_frames)
    complete_image = np.full((_total_frames, y_pix, x_pix), 0, dtype=np.uint16)
    _last_frame = 0

    for _filename in tqdm(
            range(len(_filenames)),
            total=len(_filenames),
            desc="Loading Images...",
            disable=False,
    ):
        complete_image[_last_frame:_last_frame+_num_frames[_filename], :, :] = \
            load_single_tiff(folder + "\\" + _filenames[_filename], _num_frames[_filename])
        _last_frame += _num_frames[_filename]

    return complete_image


@convert_optionals(permitted=(str, pathlib.Path), required=str, pos=0)
@validate_path(pos=0)
@if_dir_join_filename(default_name="video_meta.txt", flag_pos=0)
@validate_extension(required_extension=".txt", pos=0)
@validate_exists(pos=0)
def load_binary_meta(path: Union[str, pathlib.Path]) -> Tuple[int, int, int, str]:
    """
    Loads meta file for binary video

    :param path: The meta file (.txt ext) or directory containing metafile
    :type path: Union[str, pathlib.Path]
    :return: A tuple containing the number of frames, y pixels, and x pixels [Z x Y x X]
    :rtype: tuple[int, int, int, str]
    """
    _num_frames, _y_pixels, _x_pixels, _type = np.genfromtxt(path, delimiter=",", dtype="str")
    return int(_num_frames), int(_y_pixels), int(_x_pixels), str(_type)


@convert_optionals(permitted=(str, pathlib.Path), required=pathlib.Path, pos=0)
@validate_path(pos=0)
@validate_exists(pos=0)
def load_bruker_tiffs(folder: Union[str, pathlib.Path],
                      channels: Optional[int] = None, planes: Optional[int] = None) -> Tuple[np.ndarray]:
    """
    Load a sequence of tiff files from a directory.

    Designed to compile the outputs of a certain imaging utility
    that exports recordings such that each frame is saved as a single tiff.

    :param folder: Folder containing a sequence of single frame tiff files
    :type folder: Union[str, pathlib.Path]
    :param channels: channel to load
    :type channels: Optional[int]
    :param planes: plane to load
    :type planes: Optional[int]
    :return: complete_image:  All tiff files in the directory compiled into a single array (Z x Y x X, uint16)
    :rtype: Tuple[Any]
     """

    def load_images():
        nonlocal folder
        nonlocal tag
        nonlocal _frames
        nonlocal _y_pixels
        nonlocal _x_pixels
        nonlocal _tqdm_desc

        def load_single_image():
            nonlocal folder
            nonlocal _files
            nonlocal _file
            return np.asarray(Image.open(str(_files[_file])))

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

    _channels, _planes, _frames, _y_pixels, _x_pixels = determine_bruker_folder_contents(folder)
    pretty_print_bruker_command(_channels, _planes, _frames, _y_pixels, _x_pixels)

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
        for _channel, _plane in itertools.product(channels, planes):
            _tqdm_desc = "".join(["Loading Plane ", str(_plane),
                                  " Channel ", str(_channel), " images..."])
            tag = "".join(["Ch", str(_channel+1)])
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
    for _channel, _plane in itertools.product(channels, planes):
        _tqdm_desc = "".join(["Loading Plane ", str(_plane),
                              " Channel ", str(_channel), " images..."])
        tag = "".join(["Ch", str(_channel+1), "_00000", str(_plane+1)])
        images.append(load_images())
    images = filter(lambda image: image is not None, images)
    return tuple(images)
# REFACTOR


@validate_filename(pos=0)
@if_dir_append_filename(default_name="video_meta.txt", flag_pos=0)
@if_dir_join_filename(default_name="binary_video", flag_pos=0)
@validate_path(pos=1)
@validate_exists(pos=0)
@validate_exists(pos=1)
def load_mapped_binary(filename: str, meta_filename: Optional[str], **kwargs: str) -> np.memmap:
    """
    Loads a raw binary file in the workspace without loading into memory

    Enter the path to autofill (assumes Filename & meta are path + binary_video, video_meta.txt)

    :param filename: filename for binary video
    :type filename: str
    :param meta_filename: filename for meta file
    :type meta_filename: str
    :keyword mode: pass mode to numpy.memmap (str, default = "r")
    :return: memmap(numpy) array [Z x Y x X]
    :rtype: Any
    """

    _mode = kwargs.get("mode", "r")

    _num_frames, _y_pixels, _x_pixels, _type = load_binary_meta(meta_filename)

    return np.memmap(filename, dtype=_type, shape=(_num_frames, _y_pixels, _x_pixels), mode=_mode)
# TODO UNIT TEST FOR EXCEPTIONS


@validate_filename(pos=0)
@if_dir_append_filename(default_name="video_meta.txt", flag_pos=0)
@if_dir_join_filename(default_name="binary_video", flag_pos=0)
@validate_path(pos=1)
@validate_exists(pos=0)
@validate_exists(pos=1)
def load_raw_binary(path: str, meta_filename: Optional[str]) -> np.ndarray:
    """
    Loads a raw binary file

    Enter the path to autofill (assumes Filename & meta are path + binary_video, video_meta.txt)

    :param path: absolute filepath for binary video or directory containing a file named binary video
    :type path: str
    :param meta_filename: absolute path to meta file
    :type meta_filename: Optional[str]
    :return: numpy array [Z x Y x X]
    :rtype: Any
    """

    _num_frames, _y_pixels, _x_pixels, _type = np.genfromtxt(meta_filename, delimiter=",", dtype="str")
    _num_frames = int(_num_frames)
    _x_pixels = int(_x_pixels)
    _y_pixels = int(_y_pixels)
    return np.reshape(np.fromfile(path, dtype=_type), (_num_frames, _y_pixels, _x_pixels))
# TODO UNIT TEST FOR EXCEPTIONS


@convert_optionals(permitted=(str, pathlib.Path), required=str, pos=0)
@validate_path(pos=0)
@validate_exists(pos=0)
def load_single_tiff(filename: Union[str, pathlib.Path], num_frames: int) -> np.ndarray:
    """
    Load a single tiff file

    :param filename: absolute filename
    :param num_frames: number of frames
    :type filename: Union[str, pathlib.Path]
    :type num_frames: int
    :return: numpy array [Z x Y x X]
    :rtype: Any
    """

    return tifffile.imread(filename, key=range(0, num_frames, 1))


def pretty_print_bruker_command(channels, planes, frames, height, width) -> None:
    """
    Function simply prints the bruker folder contents detected

    :param channels: Number of channels
    :type channels: int
    :param planes: Number of planes
    :type planes: int
    :param frames: Number of frames
    :type frames: int
    :param height: Height of Image (Y Pixels)
    :type height: int
    :param width:  Width of Image (X Pixels)
    :type width:
    :rtype: None
    """
    _table = PrettyTable()
    _table.header = False
    _table.add_row(["Total Images Detected", channels * planes * frames])
    _table.add_row(["channels", channels])
    _table.add_row(["planes", planes])
    _table.add_row(["frames", frames])
    _table.add_row(["Height", height])
    _table.add_row(["Width", width])
    print("\n")
    print(_table)


@convert_optionals(permitted=(str, pathlib.Path), required=str, pos=0)
@convert_optionals(permitted=(str, pathlib.Path), required=str, pos=1)
@validate_path(pos=0)
@validate_path(pos=1)
@validate_exists(pos=0)
def repackage_bruker_tiffs(input_folder: Union[str, pathlib.Path], output_folder: Union[str, pathlib.Path],
                           *args: Union[int, tuple[int]]) -> None:
    """
    Repackages a sequence of tiff files within a directory to a smaller sequence
    of tiff stacks.
    Designed to compile the outputs of a certain imaging utility
    that exports recordings such that each frame is saved as a single tiff.

    :param input_folder: Directory containing a sequence of single frame tiff files
    :type input_folder: Union[str, pathlib.Path]
    :param output_folder: Empty directory where tiff stacks will be saved
    :type output_folder: Union[str, pathlib.Path]
    :param args: optional argument to indicate the repackaging of a specific channel and/or plane
    :type args: int
    :rtype: None
    """

    # code here is pretty rough, needs TLC. ~horror~

    def load_image():
        nonlocal input_folder
        nonlocal _files
        nonlocal _file
        nonlocal _offset
        return np.asarray(Image.open(str(_files[_file + _offset])))

    def find_files(tag: Union[str, list[str]]):
        nonlocal input_folder
        nonlocal _files

        def check_file_contents(tag_: str, file_: pathlib.WindowsPath) -> bool:
            if tag_ in str(file_.stem).split("_"):
                return True
            else:
                return False

        # reset, rather maintain code as is then make a new temporary variable since this
        # is basically instant
        _files = [_file for _file in pathlib.Path(input_folder).glob("*.tif")]

        # now find
        if isinstance(tag, list):
            tag = "".join([tag[0], "_", tag[1]])
            _files = [_file for _file in _files if tag in str(_file.stem)]
        else:
            _files = [_file for _file in _files if check_file_contents(tag, _file)]

    def not_over_4gb() -> bool:
        nonlocal _files
        nonlocal _y
        nonlocal _x

        gb = np.full((_files.__len__(), _y, _x), 1, dtype=np.uint16).nbytes
        if gb <= 3.9:  # 3.9 as a safety buffer
            return True
        else:
            return False

    _files = [_file for _file in pathlib.Path(input_folder).rglob("*.tif")]
    _channels, _planes, _frames, _y, _x = determine_bruker_folder_contents(input_folder)
    pretty_print_bruker_command(_channels, _planes, _frames, _y, _x)

    # finding the files for a specific channel/plane here
    if args:
        # unpack if necessary
        if isinstance(args[0], tuple):
            _c = args[0][0]
            _p = args[0][1]
            args = tuple([_c, _p])

        if _channels > 1 and _planes > 1 and len(args) >= 2:
            _base_tag = "00000"
            _tag = ["".join(["Ch", str(args[0])+1]), "".join([_base_tag, str(args[1]+1), ".ome"])]
            find_files(_tag)
        elif _channels == 1 and _planes > 1 and len(args) == 1:
            _base_tag = "00000"
            _tag = "".join([_base_tag, str(args[0]+1), ".ome"])
            find_files(_tag)
            print(_files)
        elif _channels == 1 and _planes > 1 and len(args) >= 2:
            _base_tag = "00000"
            _tag = "".join([_base_tag, str(args[1]+1), ".ome"])
            find_files(_tag)
        elif _channels > 1 and _planes == 1:
            _tag = "".join(["Ch", str(args[0]+1)])
            find_files(_tag)
        else:
            pass
    else:
        try:
            assert(_channels == 1)
        except AssertionError:
            print("Folder contains multiple channels")
            return
        try:
            assert(_planes == 1)
        except AssertionError:
            print("Folder contains multiple planes")
            return

    # noinspection PyTypeChecker

    # Decide whether saving in single stack is possible
    if not_over_4gb():
        _images = np.full((_frames, _y, _x), 0, dtype=np.uint16)
        _offset = 0
        for _file in range(_frames):
            _images[_file, :, :] = load_image()
        save_single_tiff(_images, "".join([output_folder, "\\compiledVideo_01_of_1.tif"]))
        return
    else:
        # noinspection PyTypeChecker
        _chunks = math.ceil(_frames/7000)
        c_idx = 1
        _offset = int()

        _pbar = tq(total=_frames)
        _pbar.set_description("Repackaging Bruker Tiffs...")

        for _chunk in range(0, _frames, 7000):

            _start_idx = _chunk
            _offset = _start_idx
            _end_idx = _chunk + 7000
            _chunk_frames = _end_idx-_start_idx
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
                save_single_tiff(image_chunk, output_folder + "\\" + "compiledVideo_0" + str(c_idx) + "_of_" +
                                 str(_chunks) + ".tif")
            else:
                save_single_tiff(image_chunk, output_folder + "\\" + "compiledVideo_" + str(c_idx) + "_of_" +
                                 str(_chunks) + ".tif")
            c_idx += 1
        _pbar.close()
    return
# REFACTOR


@convert_optionals(permitted=(str, pathlib.Path), required=str, pos=1)
@validate_path(pos=1)
@require_full_path(pos=1)
@validate_extension(required_extension=".tif", pos=1)
def save_single_tiff(images: np.ndarray, path: Union[str, pathlib.Path], type_: Optional[np.dtype] = np.uint16) -> None:
    """
    Save a numpy array to a single tiff file as type uint16

    :param images: numpy array [frames, y pixels, x pixels]
    :type images: Any
    :param path: filename or absolute path
    :type path: Union[str, pathlib.Path]
    :param type_: type for saving
    :type type_: Optional[Any]
    :rtype: None
    """

    if len(images.shape) == 2:
        with tifffile.TiffWriter(path) as tif:
            tif.save(np.floor(images).astype(type_))
        return

    with tifffile.TiffWriter(path) as tif:
        for frame in np.floor(images).astype(type_):
            tif.save(frame)


@convert_optionals(permitted=(str, pathlib.Path), required=str, pos=1)
@validate_path(pos=1)
def save_tiff_stack(images: str, output_folder: Union[str, pathlib.Path],
                    type_: Optional[np.dtype] = np.uint16) -> None:
    """
    Save a numpy array to a sequence of tiff stacks

    :param images: A numpy array containing a tiff stack [Z x Y x X]
    :type images: Any
    :param output_folder: A directory to save the sequence of tiff stacks in uint16
    :type output_folder: Union[str, pathlib.Path]
    :param type_: type for saving
    :type type_: Optional[Any]
    :rtype: None
    """
    _num_frames = images.shape[0]

    _chunks = math.ceil(_num_frames / 7000)

    c_idx = 1
    for _chunk in range(0, _num_frames, 7000):

        _start_idx = _chunk
        _end_idx = _chunk + 7000
        if _end_idx > _num_frames:
            _end_idx = _num_frames + 1

        if c_idx < 10:
            save_single_tiff(images[_start_idx:_end_idx, :, :],
                             output_folder + "\\" + "Video_0" + str(c_idx) + "_of_" + str(
                                       _chunks) + ".tif", type_)
        else:
            save_single_tiff(images[_start_idx:_end_idx, :, :],
                             output_folder + "\\" + "Video_" + str(c_idx) + "_of_" + str(
                                       _chunks) + ".tif", type_)
        c_idx += 1

    return print("Finished Saving Tiffs")


@convert_optionals(permitted=(str, pathlib.Path), required=str, pos=1)
@validate_path(pos=1)
@if_dir_append_filename(default_name="video_meta.txt", flag_pos=1)
@if_dir_join_filename(default_name="binary_video", flag_pos=1)
@validate_extension(required_extension=".txt", pos=2)
def save_raw_binary(images: np.ndarray, path: Union[str, pathlib.Path],
                    meta_filename: Optional[Union[str, pathlib.Path]]) -> None:
    """
    This function saves a tiff stack as a binary file

    :param images: Images to be saved [Z x Y x X]
    :type images: np.ndarray
    :param path:  absolute filepath for saving binary video or directory containing a file named binary video
    :type path: str
    :param meta_filename: absolute filepath for saving meta
    :type meta_filename: str
    :rtype: None
    """

    try:
        assert(pathlib.Path(path).parent.exists())
    except AssertionError:
        os.makedirs(str(pathlib.Path(path).parent))
    finally:
        with open(meta_filename, 'w') as f:
            f.writelines([str(images.shape[0]), ",", str(images.shape[1]), ",",
                          str(images.shape[2]), ",", str(images.dtype)])
    images.tofile(path)
    print("Finished saving images as a binary file.")
# TODO UNIT TEST FOR EXCEPTIONS


@convert_optionals(permitted=(str, pathlib.Path), required=str, pos=1)
@validate_path(pos=1)
@if_dir_join_filename(default_name="video.mp4", flag_pos=1)
@validate_extension(required_extension=".mp4", pos=1)
def save_video(images: np.ndarray, path: Union[str, pathlib.Path], fps: Union[float, int] = 30) -> None:
    """
    Function writes video to .mp4

    :param images: Images to be written
    :type images: Any
    :param path: Filename  (Or Complete Path)
    :type path: Union[str, pathlib.Path]
    :param fps: frame rate
    :type fps: Union[float, int]
    :rtype: None
    """

    if images.dtype.type != np.uint8:
        print("\nForcing to unsigned 8-bit\n")
        images = images.astype(np.uint8)

    print("\nWriting Images to .mp4...\n")
    mimwrite(path, images, fps=fps, quality=10, macro_block_size=4)
    print("\nFinished writing images to .mp4.\n")
# TODO: I have no unit test
