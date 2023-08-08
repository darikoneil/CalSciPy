from __future__ import annotations
from typing import Union, Optional, Mapping
from pathlib import Path
from json import load, dump

from PIL import Image
import numpy as np
import cv2
from PPVD.validation import validate_extension, validate_filename
from PPVD.parsing import convert_permitted_types_to_required

from .misc import generate_blocks, calculate_frames_per_file, zero_pad_num_to_string


"""
A collection of functions for loading, saving & converting imaging files. Stable as of version 3.5
"""


@validate_filename(pos=0)
@convert_permitted_types_to_required(permitted=(str, Path), required=Path, pos=0)
def load_binary(path: Union[str, Path],
                mapped: bool = False,
                mode: str = "r+",
                missing_metadata: Mapping = None
                ) -> Union[np.ndarray, np.memmap]:
    """
    This function loads images saved in language-agnostic binary format. Ideal for optimal read/write speeds and
    highly-robust to corruption. However, the downside is that the images and their metadata are split into two
    separate files. Images are saved with the *.bin* extension, while metadata is saved with extension *.json*.
    If for some reason you lose the metadata, you can still load the binary if you know three of the following:
    number of frames, y-pixels, x-pixels, and the datatype (:class:`numpy.dtype`)

    :param path: folder containing binary file
    :param mapped: boolean indicating whether to load image using memory-mapping
    :param mode: indicates the level of access permitted to the original binary
    :param missing_metadata: if you have lost the metadata or otherwise wish to manually provide it
    :returns: image (frames, y-pixels, x-pixels)
    """
    # If path is a folder and not just the filepath without an extension
    if not path.is_file() and not path.with_suffix(".bin").exists():
        try:
            path = path.joinpath("binary_video")
            assert (path.with_suffix(".bin").exists())
            assert (path.with_suffix(".json").exists())
        except AssertionError:
            raise FileNotFoundError(f"{path.parent.with_suffix('.bin')} or "
                                    f"{path.with_suffix('.bin')}/{path.with_suffix('.json')}")

    # add extensions
    meta_filename = path.with_suffix(".json")
    imaging_filename = path.with_suffix(".bin")

    if missing_metadata:
        metadata = _Metadata(**missing_metadata)
    else:
        metadata = _Metadata.decode(meta_filename)

    if mapped:
        return np.memmap(imaging_filename, mode=mode, dtype=metadata.dtype, shape=(metadata.frames, metadata.y,
                                                                                   metadata.x))
    else:
        images = np.fromfile(imaging_filename, dtype=metadata.dtype)
        images = np.reshape(images, (metadata.frames, metadata.y, metadata.x))
        return images


@validate_filename(pos=0)
@convert_permitted_types_to_required(permitted=(str, Path), required=Path, pos=0)
def load_images(path: Union[str, Path]) -> np.ndarray:
    """
    Load images into a numpy array. If path is a folder, all .tif files found non-recursively in the directory will be
    compiled to a single array.

    :param path: a file containing images or a folder containing several imaging stacks
    :return: numpy array (frames, y-pixels, x-pixels)
    """
    if path.is_file():
        return _load_single_tif(path)
    else:
        return _load_many_tif(path)


@validate_filename(pos=0)
@convert_permitted_types_to_required(permitted=(str, Path), required=Path, pos=0)
def save_binary(path: Union[str, Path], images: np.ndarray, name: str = "binary_video") -> int:
    """
    Save images to language-agnostic binary format. Ideal for optimal read/write speeds and highly-robust to corruption.
    However, the downside is that the images and their metadata are split into two separate files. Images are saved with
    the *.bin* extension, while metadata is saved with extension *.json*. If for some reason you lose the metadata, you
    can still load the binary if you know three of the following: number of frames, y-pixels, x-pixels, and the
    datatype. The datatype is almost always unsigned 16-bit (:class:`numpy.uint16`) for all modern imaging
    systems--even if they are collected at 12 or 13-bit.

    :param path: path to save images to. The path stem is considered the filename if it doesn't have any extension. If
    | no filename is provided then the default filename is *binary_video*.
    :param images: images to save (frames, y-pixels, x-pixels)
    :param name: specify filename for produced files
    :returns: 0 if successful
    """
    # make folder if it doesn't exist
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)

    # add extensions
    imaging_filename = path.joinpath(name).with_suffix(".bin")
    metadata_filename = path.joinpath(name).with_suffix(".json")

    # save metadata
    metadata = _Metadata(images)
    metadata.encode(metadata_filename)

    # save images
    images.tofile(imaging_filename)


@validate_filename(pos=0)
@convert_permitted_types_to_required(permitted=(str, Path), required=Path, pos=0)
def save_images(path: Union[str, Path],
                images: np.ndarray,
                name: str = "images",
                size_cap: float = 3.9,) -> int:
    """
    Save a numpy array to a single .tif file. If size > 4GB then saved as a series of files. If path is not a file and
    already exists the default filename will be *images*.

    :param path: filename or absolute path
    :param images: numpy array (frames, y pixels, x pixels)
    :param name: filename for saving images
    :param size_cap: maximum size per file
    :returns: returns 0 if successful
    """
    if not Path.exists(path):
        Path.mkdir(path, parents=True, exist_ok=True)

    file_size = images.nbytes * 1e-9  # convert to GB
    if file_size <= size_cap:  # crop at 3.9 to be saved
        filename = path.joinpath(name).with_suffix(".tif")
        _save_single_tif(filename, images)
    else:
        filename = path.joinpath(name)
        _save_many_tif(filename, images, size_cap)


@validate_extension(required_extension=".tif", pos=0)
@convert_permitted_types_to_required(permitted=(str, Path), required=str, pos=0)
def _load_single_tif(file: Union[str, Path]) -> np.ndarray:
    """
    Load a single .tif as a numpy array (implementation function)

    :param file: absolute filename
    :return: numpy array (frames, y-pixels, x-pixels)
    """
    return np.array(cv2.imreadmulti(file, flags=-1)[1])


@convert_permitted_types_to_required(permitted=(str, Path), required=Path, pos=0)
def _load_many_tif(folder: Union[str, Path]) -> np.ndarray:
    """
    Loads all .tif's within a folder into a single numpy array (implementation function)

    :param folder: folder containing a sequence of tiff stacks
    :returns: a numpy array containing the images (frames, y-pixels, x-pixels)
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
    """
    Implementation function for saving a single tif

    :param path: path to save at
    :param images: images
    :return: 0 if successful
    """
    # if single page save direct
    if len(images.shape) == 2:
        images = Image.fromarray(images)
        images.save(path)
    # if multi page iterate
    else:
        images_to_save = []
        for single_image in range(images.shape[0]):
            images_to_save.append(Image.fromarray(images[single_image, :, :]))
        images_to_save[0].save(path, format="TIFF", save_all=True, append_images=images_to_save[1:])
    return 0


@validate_extension(required_extension=".tif", pos=0)
@convert_permitted_types_to_required(permitted=(str, Path), required=Path, pos=0)
def _save_many_tif(path: Union[str, Path], images: np.ndarray, size_cap: int = 3.9) -> int:
    """
    Implementation function for saving many tif

    :param path: path to save at
    :param images: images
    :return: 0 if successful
    """
    frames_per_file = calculate_frames_per_file(*images.shape[1:], size_cap=size_cap)
    frames = list(range(images.shape[0]))
    blocks = generate_blocks(frames, frames_per_file, 0)
    idx = 0
    base_filename = path.with_suffix("")

    try:
        for block in blocks:
            str_idx = zero_pad_num_to_string(idx, 5)
            filename = base_filename.with_name("".join([base_filename.name, str_idx]))
            filename = filename.with_suffix(".tif")
            _save_single_tif(filename, images[block, :, :])
            idx += 1
    except (RuntimeError, StopIteration):
        pass
    return 0


class _Metadata:
    """
    Metadata object used for saving/loading binary images

    :ivar frames: number of frames
    :type frames: int
    :ivar y: height of each frame
    :type y: int
    :ivar x: width of each frame
    :type x: int
    :ivar dtype: datatype of each frame
    :type dtype: np.dtype
    """
    def __init__(self,
                 images: Optional[np.ndarray] = None,
                 frames: int = None,
                 y: int = None,
                 x: int = None,
                 dtype: np.dtype = None):
        """
        Metadata object used for saving/loading binary images

        :param images: images in numpy array (frames, y-pixels, x-pixels)
        :param frames: number of frames
        :param y: number of y pixels
        :param x: number of x pixels
        :param dtype: type of data
        """
        self.frames = frames
        self.y = y
        self.x = x
        self.dtype = dtype

        if images is not None:
            self.frames, self.y, self.x = images.shape
            self.dtype = str(images.dtype)

        self._validate_metadata()

    @classmethod
    def decode(cls: _Metadata, filename: Path) -> _Metadata:

        with open(str(filename), "r+") as file:
            attrs = load(file)

        return _Metadata(**attrs)

    def encode(self, filename: Path) -> int:
        meta = vars(self)
        with open(str(filename), "w+") as file:
            dump(meta, file)
        return 0

    def _validate_metadata(self) -> _Metadata:

        attrs = vars(self)

        missing_keys = [key for key in ["dtype", "frames", "y", "x"] if not attrs.get(key)]

        if "dtype" in missing_keys:
            raise AttributeError("dtype not found")

        if len(missing_keys) > 1:
            raise AttributeError("Require at least 2/3 shape keys")

        if len(missing_keys) == 1:
            setattr(self, missing_keys[0], -1)
            # set to negative one to let numpy figure it out
