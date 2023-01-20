from __future__ import annotations
import itertools
import numpy as np
import os
from PIL import Image
from tqdm.auto import tqdm
from tqdm import tqdm as tq
import tifffile
from typing import Callable, List, Tuple, Sequence, Optional, Union
import math
import pathlib
from prettytable import PrettyTable
from imageio import mimwrite

from ._validation import validate_extension, validate_filename, validate_path
from ._parsing import convert_optionals, if_dir_append_filename, if_dir_join_filename, require_full_path


@validate_path(pos=0)
@convert_optionals(permitted=(str, pathlib.Path), required=pathlib.Path)
def determine_bruker_folder_contents(ImageDirectory: Union[str, pathlib.Path]) -> Tuple[int, int, int, int, int]:
    """
    Function determine contents of the bruker folder

    :param ImageDirectory: Directory containing bruker imaging data
    :type ImageDirectory: Union[str, pathlib.Path]
    :returns: Channels, Planes, Frames, Height, Width
    :rtype: tuple
    """

    _files = [_file for _file in ImageDirectory.glob("*.tif") if _file.is_file()]

    def parser(Tag_: str, SplitStrs: list) -> str:
        return [_tag for _tag in SplitStrs if Tag_ in _tag]

    def find_num_unique_strings_given_static_substring(Tag: str) -> int:
        nonlocal _files
        _hits = [parser(Tag, str(_file).split("_")) for _file in _files]
        _hits = [_hit for _nested_hit in _hits for _hit in _nested_hit]
        return list(_hits).__len__()

    def find_num_unique_substring_containing_tag(Tag: str) -> int:
        nonlocal _files
        _hits = [parser(Tag, str(_file).split("_")) for _file in _files]
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
        nonlocal ImageDirectory
        nonlocal _files
        return np.asarray(Image.open("".join(str(_files[0])))).shape

    channels = find_channels()

    # noinspection PyTypeChecker
    return (channels, find_planes(), find_frames(), *find_dimensions())


@validate_path(pos=0)
@convert_optionals(permitted=(str, pathlib.Path), required=str, pos=0)
def load_all_tiffs(ImageDirectory: Union[str, pathlib.Path]) -> np.ndarray:
    """
    Load a sequence of tiff stacks

    :param ImageDirectory: Directory containing a sequence of tiff stacks
    :type ImageDirectory: Union[str, pathlib.Path]
    :return: complete_image numpy array [Z x Y x X] as uint16
    :rtype: Any
    """
    if isinstance(ImageDirectory, pathlib.Path):
        ImageDirectory = str(ImageDirectory)

    _fnames = [str(_fname.name) for _fname in pathlib.Path(ImageDirectory).glob("*") if ".tif" in _fname.suffix]
    y_pix, x_pix = tifffile.TiffFile(ImageDirectory + "\\" + _fnames[0]).pages[0].shape
    _num_frames = [] # initialize
    [_num_frames.append(len(tifffile.TiffFile(ImageDirectory + "\\" + _fname).pages)) for _fname in _fnames]
    _total_frames = sum(_num_frames)
    complete_image = np.full((_total_frames, y_pix, x_pix), 0, dtype=np.uint16)
    _last_frame = 0

    for _fname in tqdm(
            range(len(_fnames)),
            total=len(_fnames),
            desc="Loading Images...",
            disable=False,
    ):
        complete_image[_last_frame:_last_frame+_num_frames[_fname], :, :] = \
            load_single_tiff(ImageDirectory + "\\" + _fnames[_fname], _num_frames[_fname])
        _last_frame += _num_frames[_fname]

    return complete_image


@validate_path(pos=0)
@if_dir_join_filename(default_name="video_meta.txt", flag_pos=0)
@validate_extension(required_extension=".txt", pos=0)
def load_binary_meta(Path: str) -> Tuple[int, int, int, str]:
    """
    Loads meta file for binary video

    :param Path: The meta file (.txt ext) or directory containing metafile
    :type Path: str
    :return: A tuple containing the number of frames, y pixels, and x pixels [Z x Y x X]
    :rtype: tuple[int, int, int, str]
    """
    _num_frames, _y_pixels, _x_pixels, _type = np.genfromtxt(Path, delimiter=",", dtype="str")
    return int(_num_frames), int(_y_pixels), int(_x_pixels), str(_type)


@validate_path(pos=0)
def load_bruker_tiffs(ImageDirectory: Union[str, pathlib.Path]) -> Union[np.ndarray, Tuple[np.ndarray]]:
    """
    Load a sequence of tiff files from a directory.

    Designed to compile the outputs of a certain imaging utility
    that exports recordings such that each frame is saved as a single tiff.

    :param ImageDirectory: Directory containing a sequence of single frame tiff files
    :type ImageDirectory: Union[str, pathlib.Path]
    :return: complete_image:  All tiff files in the directory compiled into a single array (Z x Y x X, uint16)
    :rtype: Any
     """

    _channels, _planes, _frames, _y_pixels, _x_pixels = determine_bruker_folder_contents(ImageDirectory)
    pretty_print_bruker_command(_channels, _planes, _frames, _y_pixels, _x_pixels)

    def find_files(Tag: Union[str, list[str]]):
        nonlocal ImageDirectory
        nonlocal _files

        def check_file_contents(Tag_: str, File_: pathlib.WindowsPath) -> bool:
            if Tag_ in str(File_).split("_"):
                return True
            else:
                return False

        # reset, rather maintain code as is then make a new temporary variable since this
        # is basically instant
        _files = [_file for _file in pathlib.Path(ImageDirectory).rglob("*.tif")]

        # now find
        if isinstance(Tag, list):
            Tag = "".join([Tag[0], Tag[1]])
            _files = [_file for _file in _files if Tag in str(_file.stem)]
        else:
            _files = [_file for _file in _files if check_file_contents(Tag, _file)]

    def load_images():
        nonlocal _files
        nonlocal _frames
        nonlocal _y_pixels
        nonlocal _x_pixels
        nonlocal _tqdm_desc

        def load_image():
            nonlocal ImageDirectory
            nonlocal _files
            nonlocal _file
            return np.asarray(Image.open(str(_files[_file])))

        # Bruker is usually saved to .tif in 0-65536 (uint16) even though recording says 8192 (uint13)
        complete_image = np.full((_frames, _y_pixels, _x_pixels), 0, dtype=np.uint16)
        for _file in tqdm(
                range(_frames),
                total=_frames,
                desc=_tqdm_desc,
                disable=False,
        ):
            complete_image[_file, :, :] = load_image()

        return complete_image

    _files = [_file for _file in pathlib.Path(ImageDirectory).rglob("*.tif")]

    if _channels == 1 and _planes == 1:
        _tqdm_desc = "Loading Images..."
        return load_images()
    elif _channels > 1 and _planes == 1:
        images = []
        for _channel in range(_channels):
            _tqdm_desc = "".join(["Loading Channel ", str(_channel+1), " Images..."])
            _tag = "".join(["Ch", str(_channel+1)])
            find_files(_tag)
            images.append(load_images())
        return images
    elif _channels == 1 and _planes > 1:
        images = []
        _base_tag = "00000"
        for _plane in range(_planes):
            _tqdm_desc = "".join(["Loading Plane ", str(_plane), " Images..."])
            _tag = "".join([_base_tag, str(_plane+1), ".ome"])
            find_files(_tag)
            images.append(load_images())
        return images
    elif _channels > 1 and _planes > 1:
        images = []
        _base_tag = "00000"
        for _channel, _plane in itertools.combinations(range(_channels), range(_planes)):
            _tqdm_desc = "".join(["Loading Plane ", str(_plane),
                                  " Channel ", str(_channel), " Images..."])
            _tag = ["".join(["Ch", str(_channel)]), "".join([_base_tag, str(_plane+1), ".ome"])]
            find_files(_tag)
            images.append(load_images())
        return images
# FIXME: I am broken on multi-plane/multi-channel images


@validate_filename(pos=0)
@if_dir_append_filename(default_name="video_meta.txt", flag_pos=0)
@if_dir_join_filename(default_name="binary_video", flag_pos=0)
@validate_path(pos=1)
def load_mapped_binary(Filename: str, MetaFile: Optional[str], **kwargs: str) -> np.memmap:
    """
    Loads a raw binary file in the workspace without loading into memory

    Enter the path to autofill (assumes Filename & meta are path + binary_video, video_meta.txt)

    :param Filename: filename for binary video
    :type Filename: str
    :param MetaFile: filename for meta file
    :type MetaFile: str
    :keyword mode: pass mode to numpy.memmap (str, default = "r")
    :return: memmap(numpy) array [Z x Y x X]
    :rtype: Any
    """

    _mode = kwargs.get("mode", "r")

    _num_frames, _y_pixels, _x_pixels, _type = load_binary_meta(MetaFile)

    return np.memmap(Filename, dtype=_type, shape=(_num_frames, _y_pixels, _x_pixels), mode=_mode)


@validate_filename(pos=0)
@if_dir_append_filename(default_name="video_meta.txt", flag_pos=0)
@if_dir_join_filename(default_name="binary_video", flag_pos=0)
@validate_path(pos=1)
def load_raw_binary(Path: str, MetaFile: Optional[str]) -> np.ndarray:
    """
    Loads a raw binary file

    Enter the path to autofill (assumes Filename & meta are path + binary_video, video_meta.txt)

    :param Path: absolute filepath for binary video or directory containing a file named binary video
    :type Path: str
    :param MetaFile: absolute path to meta file
    :type MetaFile: Optional[str]
    :return: numpy array [Z x Y x X]
    :rtype: Any
    """

    _num_frames, _y_pixels, _x_pixels, _type = np.genfromtxt(MetaFile, delimiter=",", dtype="str")
    _num_frames = int(_num_frames)
    _x_pixels = int(_x_pixels)
    _y_pixels = int(_y_pixels)
    return np.reshape(np.fromfile(Path, dtype=_type), (_num_frames, _y_pixels, _x_pixels))


@validate_path(pos=0)
@convert_optionals(permitted=(str, pathlib.Path), required=str, pos=0)
def load_single_tiff(Filename: Union[str, pathlib.Path], NumFrames: int) -> np.ndarray:
    """
    Load a single tiff file

    :param Filename: absolute filename
    :param NumFrames: number of frames
    :type Filename: Union[str, pathlib.Path]
    :type NumFrames: int
    :return: numpy array [Z x Y x X]
    :rtype: Any
    """

    return tifffile.imread(Filename, key=range(0, NumFrames, 1))


def pretty_print_bruker_command(Channels, Planes, Frames, Height, Width) -> None:
    """
    Function simply prints the bruker folder contents detected

    :param Channels: Number of Channels
    :type Channels: int
    :param Planes: Number of Planes
    :type Planes: int
    :param Frames: Number of Frames
    :type Frames: int
    :param Height: Height of Image (Y Pixels)
    :type Height: int
    :param Width:  Width of Image (X Pixels)
    :type Width:
    :rtype: None
    """
    _table = PrettyTable()
    _table.header = False
    _table.add_row(["Total Images Detected", Channels * Planes * Frames])
    _table.add_row(["Channels", Channels])
    _table.add_row(["Planes", Planes])
    _table.add_row(["Frames", Frames])
    _table.add_row(["Height", Height])
    _table.add_row(["Width", Width])
    print("\n")
    print(_table)


@validate_path(pos=0)
@validate_path(pos=1)
@convert_optionals(permitted=(str, pathlib.Path), required=str, pos=0)
@convert_optionals(permitted=(str, pathlib.Path), required=str, pos=1)
def repackage_bruker_tiffs(ImageDirectory: Union[str, pathlib.Path], OutputDirectory: Union[str, pathlib.Path],
                           *args: Union[int, tuple[int]]) -> None:
    """
    Repackages a sequence of tiff files within a directory to a smaller sequence
    of tiff stacks.
    Designed to compile the outputs of a certain imaging utility
    that exports recordings such that each frame is saved as a single tiff.

    :param ImageDirectory: Directory containing a sequence of single frame tiff files
    :type ImageDirectory: Union[str, pathlib.Path]
    :param OutputDirectory: Empty directory where tiff stacks will be saved
    :type OutputDirectory: Union[str, pathlib.Path]
    :param args: optional argument to indicate the repackaging of a specific channel and/or plane
    :type args: int
    :rtype: None
    """

    # code here is pretty rough, needs TLC. Maybe simpler to use a generator to generate chunks

    def load_image():
        nonlocal ImageDirectory
        nonlocal _files
        nonlocal _file
        nonlocal _offset
        return np.asarray(Image.open(str(_files[_file + _offset])))

    def find_files(Tag: Union[str, list[str]]):
        nonlocal ImageDirectory
        nonlocal _files

        def check_file_contents(Tag_: str, File_: pathlib.WindowsPath) -> bool:
            if Tag_ in str(File_.stem).split("_"):
                return True
            else:
                return False

        # reset, rather maintain code as is then make a new temporary variable since this
        # is basically instant
        _files = [_file for _file in pathlib.Path(ImageDirectory).glob("*.tif")]

        # now find
        if isinstance(Tag, list):
            Tag = "".join([Tag[0], "_", Tag[1]])
            _files = [_file for _file in _files if Tag in str(_file.stem)]
        else:
            _files = [_file for _file in _files if check_file_contents(Tag, _file)]

    def not_over_4gb() -> bool:
        nonlocal _files
        nonlocal _y
        nonlocal _x

        gb = np.full((_files.__len__(), _y, _x), 1, dtype=np.uint16).nbytes
        if gb <= 3.9: # 3.9 as a safety buffer
            return True
        else:
            return False

    _files = [_file for _file in pathlib.Path(ImageDirectory).rglob("*.tif")]
    _channels, _planes, _frames, _y, _x = determine_bruker_folder_contents(ImageDirectory)
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
        save_single_tiff(_images, "".join([OutputDirectory, "\\compiledVideo_01_of_1.tif"]))
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
                save_single_tiff(image_chunk, OutputDirectory + "\\" + "compiledVideo_0" + str(c_idx) + "_of_" + str(_chunks) + ".tif")
            else:
                save_single_tiff(image_chunk, OutputDirectory + "\\" + "compiledVideo_" + str(c_idx) + "_of_" + str(_chunks) + ".tif")
            c_idx += 1
        _pbar.close()
    return


@validate_filename(pos=1)
@require_full_path(pos=1)
@convert_optionals(permitted=(str, pathlib.Path), required=str, pos=1)
@validate_extension(required_extension=".tif", pos=1)
def save_single_tiff(Images: np.ndarray, Path: Union[str, pathlib.Path], Type: Optional[np.dtype] = np.uint16) -> None:
    """
    Save a numpy array to a single tiff file as type uint16

    :param Images: numpy array [frames, y pixels, x pixels]
    :type Images: Any
    :param Path: filename or absolute path
    :type Path: Union[str, pathlib.Path]
    :param Type: type for saving
    :type Type: Optional[Any]
    :rtype: None
    """

    if len(Images.shape) == 2:
        with tifffile.TiffWriter(Path) as tif:
            tif.save(np.floor(Images).astype(Type))
        return

    with tifffile.TiffWriter(Path) as tif:
        for frame in np.floor(Images).astype(Type):
            tif.save(frame)


@validate_path(pos=1)
@convert_optionals(permitted=(str, pathlib.Path), required=str, pos=1)
def save_tiff_stack(Images: str, OutputDirectory: Union[str, pathlib.Path], Type: Optional[np.dtype] = np.uint16) -> None:
    """
    Save a numpy array to a sequence of tiff stacks

    :param Images: A numpy array containing a tiff stack [Z x Y x X]
    :type Images: Any
    :param OutputDirectory: A directory to save the sequence of tiff stacks in uint16
    :type OutputDirectory: Union[str, pathlib.Path]
    :param Type: type for saving
    :type Type: Optional[Any]
    :rtype: None
    """
    _num_frames = Images.shape[0]

    _chunks = math.ceil(_num_frames / 7000)

    c_idx = 1
    for _chunk in range(0, _num_frames, 7000):

        _start_idx = _chunk
        _end_idx = _chunk + 7000
        if _end_idx > _num_frames:
            _end_idx = _num_frames + 1

        if c_idx < 10:
            save_single_tiff(Images[_start_idx:_end_idx, :, :],
                                    OutputDirectory + "\\" + "Video_0" + str(c_idx) + "_of_" + str(
                                       _chunks) + ".tif")
        else:
            save_single_tiff(Images[_start_idx:_end_idx, :, :],
                                    OutputDirectory + "\\" + "Video_" + str(c_idx) + "_of_" + str(
                                       _chunks) + ".tif")
        c_idx += 1

    return print("Finished Saving Tiffs")


@validate_path(pos=1)
@convert_optionals(permitted=(str, pathlib.Path), required=str, pos=1)
@if_dir_append_filename(default_name="video_meta.txt", flag_pos=1)
@if_dir_join_filename(default_name="binary_video", flag_pos=1)
@validate_extension(required_extension=".txt", pos=2)
def save_raw_binary(Images: np.ndarray, Path: Union[str, pathlib.Path], MetaFile: Optional[Union[str, pathlib.Path]]) -> None:
    """
    This function saves a tiff stack as a binary file

    :param Images: Images to be saved [Z x Y x X]
    :type Images: np.ndarray
    :param Path:  absolute filepath for saving binary video or directory containing a file named binary video
    :type Path: str
    :param MetaFile: absolute filepath for saving meta
    :type MetaFile: str
    :rtype: None
    """

    try:
        assert(pathlib.Path(Path).parent.exists())
    except AssertionError:
        os.makedirs(str(pathlib.Path(Path).parent))
    finally:
        with open(MetaFile, 'w') as f:
            f.writelines([str(Images.shape[0]), ",", str(Images.shape[1]), ",",
                          str(Images.shape[2]), ",", str(Images.dtype)])
    Images.tofile(Path)
    print("Finished saving images as a binary file.")


@validate_path(pos=1)
@convert_optionals(permitted=(str, pathlib.Path), required=str, pos=1)
@if_dir_join_filename(default_name="video.mp4", flag_pos=1)
@validate_extension(required_extension=".mp4", pos=1)
def save_video(Images: np.ndarray, Path: Union[str, pathlib.Path], fps: Union[float, int] = 30) -> None:
    """
    Function writes video to .mp4

    :param Images: Images to be written
    :type Images: Any
    :param Path: Filename  (Or Complete Path)
    :type Path: Union[str, pathlib.Path]
    :param fps: frame rate
    :type fps: Union[float, int]
    :rtype: None
    """

    if Images.dtype.type != np.uint8:
        print("\nForcing to unsigned 8-bit\n")
        Images = Images.astype(np.uint8)

    print("\nWriting Images to .mp4...\n")
    mimwrite(Path, Images, fps=fps, quality=10, macro_block_size=4)
    print("\nFinished writing images to .mp4.\n")
# TODO: I have an incomplete unit test
