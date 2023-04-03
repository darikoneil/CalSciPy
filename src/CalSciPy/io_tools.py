from __future__ import annotations
from typing import Union
from pathlib import Path
from math import floor
import cv2
import numpy as np
from json_tricks import load, dump
from PPVD.validation import validate_extension, validate_filename
from PPVD.parsing import convert_permitted_types_to_required
from CalSciPy.misc import generate_blocks


@convert_permitted_types_to_required(permitted=(str, Path), required=Path, pos=0)
def load_binary(path: Union[str, Path], mapped: bool = False) -> Union[np.ndarray, np.memmap]:
    if not path.is_file():
        path = path.joinpath("binary_video")

    metadata = _load_binary_meta(path)
    filename = path.with_suffix(".bin")

    if mapped:
        return np.memmap(filename, mode="r+", dtype=metadata.dtype, shape=(metadata.frames, metadata.y, metadata.x))
    else:
        images = np.fromfile(filename, dtype=metadata.dtype)
        images = np.reshape(images, (metadata.frames, metadata.y, metadata.x))
        return images


@validate_filename(pos=0)
@convert_permitted_types_to_required(permitted=(str, Path), required=Path, pos=0)
def load_images(path: Union[str, Path]) -> np.ndarray:
    """
    Load images into a numpy array. If path is a folder, all .tif files found non-recursively in the directory will be
    compiled to a single array

    :param path: a file containing images or a folder containing several imaging stacks
    :type path: str or pathlib.Path
    :return: numpy array (frames, y-pixels, x-pixels)
    :rtype: numpy.ndarray
    """
    if path.is_file():
        return _load_single_tif(path)
    else:
        return _load_many_tif(path)


@convert_permitted_types_to_required(permitted=(str, Path), required=Path, pos=0)
def save_binary(path: Union[str, Path], images: np.ndarray) -> int:
    """
    Save images to language-agnostic binary format. Ideal for optimal read/write speeds and highly-robust to corruption.
    However, the downside is that the images and their metadata are split into two separate files. Images are saved with
    the *.bin* extension, while metadata is saved with extension *.json*. If for some reason you lose the metadata, you
    can still load the binary if you know three of the following: number of frames, y-pixels, x-pixels, and the
    datatype. The datatype is almost always unsigned 16-bit for all modern imaging systems--even if they are collected
    at 12 or 13-bit.

    :param path: path to save images to. The path stem is considered the filename if it doesn't exist. If no filename is
        provided then the default filename is *binary_video*.
    :type path: str or pathlib.Path
    :param images: images to save (frames, y-pixels, x-pixels)
    :type images: numpy.ndarray
    :return: 0 if successful
    :rtype: int
    """
    # parse the desired path
    if path.is_file():
        name = Path.name
        file_path = Path(path.parents)
    else:
        name = "binary_video"
        file_path = path

    if not file_path.exists():
        file_path.mkdir(parents=True, exist_ok=True)

    # add extensions
    imaging_filename = file_path.joinpath(name).with_suffix(".bin")
    metadata_filename = file_path.joinpath(name).with_suffix(".json")

    # save metadata
    metadata = _Metadata(images)
    dump(metadata, str(metadata_filename))

    # save images
    images.tofile(str(imaging_filename))


@validate_filename(pos=0)
@convert_permitted_types_to_required(permitted=(str, Path), required=Path, pos=0)
def save_images(path: Union[str, Path], images: np.ndarray, size_cap: float = 3.9) -> int:
    """
    Save a numpy array to a single .tif file. If size > 4GB then saved as a series of files. If path is not a file and
    already exists the default filename will be *images*.

    :param path: filename or absolute path
    :type path: str or pathlib.Path
    :param images: numpy array (frames, y pixels, x pixels)
    :type images: numpy.ndarray
    :param size_cap: maximum size per file
    :type size_cap: float = 3.9
    :return: returns 0 if successful
    :rtype: int
    """
    # parse desired path
    if Path.exists(path):
        if path.is_file():
            name = path.name
            file_path = path.parent
        else:
            name = "images"
            file_path = path
    else:
        if path.is_file():
            name = path.name
            file_path = path.parent
        else:
            name = "images"
            file_path = path

    if not Path.exists(file_path):
        Path.mkdir(file_path, parents=True, exist_ok=True)

    file_size = images.nbytes * 1e-9  # convert to GB
    if file_size <= size_cap:  # crop at 3.9 to be saved
        filename = file_path.joinpath(name).with_suffix(".tif")
        _save_single_tif(filename, images)
    else:
        filename = file_path.joinpath(name)
        _save_many_tif(filename, images, size_cap)


@convert_permitted_types_to_required(permitted=(str, Path), required=Path, pos=0)
def _load_binary_meta(path: Union[str, Path]) -> _Metadata:
    path = path.with_suffix(".json")
    return load(str(path))


@validate_extension(required_extension=".tif", pos=0)
@convert_permitted_types_to_required(permitted=(str, Path), required=str, pos=0)
def _load_single_tif(file: Union[str, Path]) -> np.ndarray:
    """
    Load a single .tif as a numpy array

    :param file: absolute filename
    :type file: str or pathlib.Path
    :return: numpy array (frames, y-pixels, x-pixels)
    :rtype: numpy.ndarray
    """
    return np.array(cv2.imreadmulti(file, flags=-1)[1])


@convert_permitted_types_to_required(permitted=(str, Path), required=Path, pos=0)
def _load_many_tif(folder: Union[str, Path]) -> np.ndarray:
    """
    Loads all .tif's within a folder into a single numpy array.

    :param folder: folder containing a sequence of tiff stacks
    :type folder: str or pathlib.Path
    :return: a numpy array containing the images (frames, y-pixels, x-pixels)
    :rtype: numpy.ndarray
    """
    files = list(folder.glob("*tif"))
    images = [_load_single_tif(file) for file in files]

    for image in images:
        assert (image.shape[1:] == images[0].shape[1:]), "Images don't maintain consistent shape"

    for image in images:
        assert (image.dtype == images[0].dtype), "Images don't maintain consistent bit depth"

    return np.concatenate(images, axis=0)


@validate_extension(required_extension=".tif", pos=0)
@convert_permitted_types_to_required(permitted=(str, Path), required=str, pos=0)
def _save_single_tif(path: Union[str, Path], images: np.ndarray) -> int:
    cv2.imwritemulti(path, images)


@validate_extension(required_extension=".tif", pos=0)
@convert_permitted_types_to_required(permitted=(str, Path), required=Path, pos=0)
def _save_many_tif(path: Union[str, Path], images: np.ndarray, size_cap: int = 3.9) -> int:
    single_frame_size = images[0, :, :].nbytes * 1e-9
    frames_per_file = floor(size_cap / single_frame_size)
    frames = list(range(images.shape[0]))
    blocks = generate_blocks(frames, frames_per_file, 0)
    idx = 0
    base_filename = path.with_suffix("")

    try:
        for block in blocks:
            if idx <= 9:
                filename = base_filename.with_name("".join([base_filename.name, "_0", str(idx)]))
                filename = filename.with_suffix(".tif")
            else:
                filename = path.with_name("".join([base_filename.name, "_", str(idx)]))
                filename = filename.with_suffix(".tif")
            _save_single_tif(filename, images[block, :, :])
            idx += 1
    except RuntimeError:
        pass
    return 0


class _Metadata:
    def __init__(self, images: np.ndarray):
        """
        Metadata object using for saving/loading binary images

        :param images: images in numpy array (frames, y-pixels, x-pixels)
        :type images: numpy.ndarray
        """
        self.frames, self.y, self.x = images.shape
        self.dtype = str(images.dtype)
