from __future__ import annotations
from typing import Union
from pathlib import Path
from json_tricks import load, dump
import numpy as np
from PIL import Image
from PPVD.parsing import convert_permitted_types_to_required


@convert_permitted_types_to_required(permitted=(str, Path), required=Path, pos=0)
def load_images(path: Union[str, Path]) -> np.ndarray:
    """
    Load images into a numpy array. If path is a folder, all .tif files found non-recursively in the directory will be
    compiled to a single array

    :param path: a file containing images or a folder containg several imaging stacks
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
    datatype. The datatype is almost always usigned 16-bit for all modern imaging systems--even if they are collected at
    12 or 13-bit.

    :param path: path to save images to. The path stem is considered the filename if it doesn't exist. If no filename is
        provided then the default filename is *binaryvideo*.
    :type path: str or pathlib.Path
    :param images: images to save (frames, y-pixels, x-pixels)
    :type images: numpy.ndarray
    :return: 0 if successful
    :rtype: int
    """
    # parse the desired path
    if not Path.exists(path):
        name = Path.stem
        file_path = path.parents
    else:
        name = path.joinpath("binary_video")
        file_path = path

    # add extensions
    imaging_filename = file_path.joinpath(name).with_suffix(".bin")
    metadata_filename = file_path.joinpath(name).with_suffix(".json")

    # save metadata
    metadata = _Metadata(images)
    dump(metadata, metadata_filename)

    # save images
    images.tofile(imaging_filename)


def _load_binary_meta(path: Union[str, Path]) -> Tuple[int, int, int, str]:
    return load(path)


@convert_permitted_types_to_required(permitted=(str, Path), required=str, pos=0)
def _load_single_tif(file: Union[str, Path]) -> np.ndarray:
    """
    Load a single .tif as a numpy array

    :param file: absolute filename
    :type file: str or pathlib.Path
    :return: numpy array (frames, y-pixels, x-pixels)
    :rtype: numpy.ndarray
    """
    return np.array(Image.open(file))


@convert_permitted_types_to_required(permitted=(str, Path), required=Path, pos=0)
def _load_many_tif(folder: Union[str, Path]) -> np.ndarray:
    """
    Loads all .tif's within a folder into a single numpy array.

    :param folder: folder containing a sequence of tiff stacks
    :type folder: str or pathlib.Path
    :return: a numpy array containing the images (frames, y-pixels, x-pixels)
    :rtype: numpy.ndarray
    """
    files = [file for file in folder.glob("*.tif")]
    images = [_load_single_tif(file) for file in files]

    for image in images:
        assert (image.shape[1:] == images[0].shape[1:]), "Images don't maintain consistent shape"

    for image in images:
        assert (image.dtype == images[0].dtype), "Images don't maintain consistent bit depth"

    return np.concatenate(images, axis=0)


class _Metadata:
    def __init__(self, images: np.ndarray):
        """
        Metadata object using for saving/loading binary images

        :param images: images in numpy array (frames, y-pixels, x-pixels)
        :type images: numpy.ndarray
        """
        self.frames, self.y, self.x = images.shape
        self.dtype = images.dtype
