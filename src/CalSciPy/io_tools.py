from __future__ import annotations
import numpy as np
import os
from tqdm.auto import tqdm
from tqdm import tqdm as tq
import tifffile
from typing import Tuple, Optional, Union
import math
import pathlib
from imageio import mimwrite

from PPVD.validation import validate_exists, validate_extension, validate_filename, validate_path
from PPVD.parsing import convert_permitted_types_to_required, if_dir_append_filename, if_dir_join_filename, \
    require_full_path, find_num_unique_files_given_static_substring, find_num_unique_files_containing_tag

#
@convert_permitted_types_to_required(permitted=(str, pathlib.Path), required=str, pos=0)
@validate_exists(pos=0)
def load_all_tiffs(folder: Union[str, pathlib.Path]) -> np.ndarray:
    """
    Loads all .tif's within a folder into a single numpy array. Assumes .tif files are the standard unsigned 16-bit
    exported by the majority (all?) of imaging software.

    :param folder: folder containing a sequence of tiff stacks
    :type folder: str or pathlib.Path
    :return: a numpy array containing the images (frames, y-pixels, x-pixels)
    :rtype: numpy.ndarray
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
            load_single_tiff(folder + "\\" + _filenames[_filename])
        _last_frame += _num_frames[_filename]

    return complete_image


@convert_permitted_types_to_required(permitted=(str, pathlib.Path), required=str, pos=0)
@if_dir_join_filename(default_name="video_meta.txt", flag_pos=0)
@validate_extension(required_extension=".txt", pos=0)
@validate_exists(pos=0)
def load_binary_meta(path: Union[str, pathlib.Path]) -> Tuple[int, int, int, str]:
    """
    Loads the meta file for an associated binary video

    :param path: The meta file (.txt ext) or a directory containing metafile
    :type path: str or pathlib.Path
    :return: A tuple where (frames, y-pixels, x-pixels, :class:`numpy.dtype`)
    :rtype: tuple[int, int, int, str]
    """
    _num_frames, _y_pixels, _x_pixels, _type = np.genfromtxt(path, delimiter=",", dtype="str")
    return int(_num_frames), int(_y_pixels), int(_x_pixels), str(_type)


@convert_permitted_types_to_required(permitted=(str, pathlib.Path), required=str, pos=0)
@validate_filename(pos=0)
@if_dir_append_filename(default_name="video_meta.txt", flag_pos=0)
@if_dir_join_filename(default_name="binary_video", flag_pos=0)
@validate_exists(pos=0)
@validate_exists(pos=1)
def load_mapped_binary(path: Union[str, pathlib.Path], meta_filename: Optional[str] = None, **kwargs: str) -> np.memmap:
    """
    Loads a raw binary file as numpy array without loading into memory (memmap). Enter a directory to autogenerate the
    default filenames (binary_video, video_meta.txt)

    :param path: absolute filepath for binary video or a folder containing a binary video with the default filename
    :type path: str or pathlib.Path
    :param meta_filename: absolute path to meta file
    :type meta_filename: Optional[str] = None
    :keyword mode: mode used in loading numpy.memmap (str, default = "r")
    :return: memmap (numpy) array (frames, y-pixels, x-pixels)
    :rtype: numpy.memmap
    """

    _mode = kwargs.get("mode", "r")

    _num_frames, _y_pixels, _x_pixels, _type = load_binary_meta(meta_filename)

    return np.memmap(path, dtype=_type, shape=(_num_frames, _y_pixels, _x_pixels), mode=_mode)
# TODO UNIT TEST FOR EXCEPTIONS


@convert_permitted_types_to_required(permitted=(str, pathlib.Path), required=str, pos=0)
@validate_filename(pos=0)
@if_dir_append_filename(default_name="video_meta.txt", flag_pos=0)
@if_dir_join_filename(default_name="binary_video", flag_pos=0)
@validate_exists(pos=0)
@validate_exists(pos=1)
def load_raw_binary(path: Union[str, pathlib.Path], meta_filename: Optional[str] = None) -> np.ndarray:
    """
    Loads a raw binary file as a numpy array. Enter a directory to autogenerate the default
    filenames (binary_video, video_meta.txt)

    :param path: absolute filepath for binary video or directory containing a file named binary video
    :type path: str or pathlib.Path
    :param meta_filename: absolute path to meta file
    :type meta_filename: Optional[str] = None
    :return: numpy array (frames, y-pixels, x-pixels)
    :rtype: numpy.ndarray
    """

    _num_frames, _y_pixels, _x_pixels, _type = np.genfromtxt(meta_filename, delimiter=",", dtype="str")
    _num_frames = int(_num_frames)
    _x_pixels = int(_x_pixels)
    _y_pixels = int(_y_pixels)
    return np.reshape(np.fromfile(path, dtype=_type), (_num_frames, _y_pixels, _x_pixels))
# TODO UNIT TEST FOR EXCEPTIONS


@convert_permitted_types_to_required(permitted=(str, pathlib.Path), required=str, pos=0)
@validate_exists(pos=0)
def load_single_tiff(path: Union[str, pathlib.Path]) -> np.ndarray:
    """
    Load a single .tif as a numpy array

    :param path: absolute filename
    :type path: str or pathlib.Path
    :return: numpy array (frames, y-pixels, x-pixels)
    :rtype: numpy.ndarray
    """
    image = []
    with tifffile.TiffFile(path) as _file:
        for _frame in _file.pages:
            image.append(_frame.asarray())

    return np.array(image)


@convert_permitted_types_to_required(permitted=(str, pathlib.Path), required=str, pos=1)
@require_full_path(pos=1)
@validate_extension(required_extension=".tif", pos=1)
def save_single_tiff(images: np.ndarray, path: Union[str, pathlib.Path], type_: Optional[np.dtype] = np.uint16) -> None:
    """
    Save a numpy array to a single .tif file. Size must be <4 GB.

    :param images: numpy array [frames, y pixels, x pixels]
    :type images: numpy.ndarray
    :param path: filename or absolute path
    :type path: str or pathlib.Path
    :param type_: type for saving
    :type type_: Optional[numpy.dtype] = numpy.uint16
    :rtype: None
    """

    # just save if single frame
    if len(images.shape) == 2:
        with tifffile.TiffWriter(path) as tif:
            tif.save(np.floor(images).astype(type_))
        return

    with tifffile.TiffWriter(path) as tif:
        for frame in np.floor(images).astype(type_):
            tif.save(frame)


@convert_permitted_types_to_required(permitted=(str, pathlib.Path), required=str, pos=1)

def save_tiff_stack(images: str, output_folder: Union[str, pathlib.Path],
                    type_: Optional[np.dtype] = np.uint16) -> None:
    """
    Save a numpy array to a sequence of .tif stacks

    :param images: numpy array (frames, y-pixels, x-pixels)
    :type images: numpy.ndarray
    :param output_folder: folder to save the sequence of .tif stacks
    :type output_folder: str or pathlib.Path
    :param type_: type for saving
    :type type_: Optional[numpy.dtype] = numpy.uint16
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
# REFACTOR at some point the CHUNKING


@convert_permitted_types_to_required(permitted=(str, pathlib.Path), required=str, pos=1)
@if_dir_append_filename(default_name="video_meta.txt", flag_pos=1)
@if_dir_join_filename(default_name="binary_video", flag_pos=1)
@validate_extension(required_extension=".txt", pos=2)
def save_raw_binary(images: np.ndarray, path: Union[str, pathlib.Path],
                    meta_filename: Optional[Union[str, pathlib.Path]]) -> None:
    """
    Save a numpy array as a binary file with an associated meta .txt file

    :param images: numpy array (frames, y-pixels, x-pixels)
    :type images: numpy.ndarray
    :param path:  folder to save in or an absolute filepath for binary video file
    :type path: str
    :param meta_filename: absolute filepath for saving meta .txt file
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


@convert_permitted_types_to_required(permitted=(str, pathlib.Path), required=str, pos=1)
@if_dir_join_filename(default_name="video.mp4", flag_pos=1)
@validate_extension(required_extension=".mp4", pos=1)
def save_video(images: np.ndarray, path: Union[str, pathlib.Path], fps: float = 30.0) -> None:
    """
    Save numpy array as an .mp4. Will be converted to uint8 if any other datatype.

    :param images: numpy array (frames, y-pixels, x-pixels)
    :type images: numpy.array
    :param path: absolute filepath or filename
    :type path: str or pathlib.Path
    :param fps: frame rate for saved video
    :type fps: float = 30.0
    :rtype: None
    """

    if images.dtype.type != np.uint8:
        print("\nForcing to unsigned 8-bit\n")
        images = images.astype(np.uint8)

    print("\nWriting Images to .mp4...\n")
    mimwrite(path, images, fps=fps, quality=10, macro_block_size=4)
    print("\nFinished writing images to .mp4.\n")
# TODO: NO UNIT TEST
