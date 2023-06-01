from __future__ import annotations
from typing import Union, Optional
from pathlib import Path
from json import load, dump


from PIL import Image
import numpy as np
import cv2
from PPVD.validation import validate_extension, validate_filename
from PPVD.parsing import convert_permitted_types_to_required


from .misc import generate_blocks, calculate_frames_per_file


@convert_permitted_types_to_required(permitted=(str, Path), required=Path, pos=0)
def load_binary(path: Union[str, Path], mapped: bool = False) -> Union[np.ndarray, np.memmap]:
    """
    This function loads images saved in language-agnostic binary format. Ideal for optimal read/write speeds and
    highly-robust to corruption. However, the downside is that the images and their metadata are split into two
    separate files. Images are saved with the *.bin* extension, while metadata is saved with extension *.json*.
    If for some reason you lose the metadata, you can still load the binary if you know three of the following:
    number of frames, y-pixels, x-pixels, and the datatype (:class:`numpy.dtype`)

    :param path: folder containing binary file
    :param mapped: boolean indicating whether to load image using memory-mapping
    :returns: image (frames, y-pixels, x-pixels)
    """
    if not path.is_file():
        path = path.joinpath("binary_video")

    # add extensions
    meta_filename = path.with_suffix(".json")
    imaging_filename = path.with_suffix(".bin")

    metadata = _Metadata.decode(meta_filename)

    if mapped:
        return np.memmap(imaging_filename, mode="r+", dtype=metadata.dtype, shape=(metadata.frames, metadata.y,
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


@convert_permitted_types_to_required(permitted=(str, Path), required=Path, pos=0)
def save_binary(path: Union[str, Path], images: np.ndarray) -> int:
    """
    Save images to language-agnostic binary format. Ideal for optimal read/write speeds and highly-robust to corruption.
    However, the downside is that the images and their metadata are split into two separate files. Images are saved with
    the *.bin* extension, while metadata is saved with extension *.json*. If for some reason you lose the metadata, you
    can still load the binary if you know three of the following: number of frames, y-pixels, x-pixels, and the
    datatype. The datatype is almost always unsigned 16-bit (:class:`numpy.uint16`) for all modern imaging
    systems--even if they are collected at 12 or 13-bit.

    :param path: path to save images to. The path stem is considered the filename if it doesn't have any extension. If no filename is provided then the default filename is *binary_video*.
    :param images: images to save (frames, y-pixels, x-pixels)
    :returns: 0 if successful
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
    metadata.encode(metadata_filename)

    # save images
    images.tofile(imaging_filename)


@validate_filename(pos=0)
@convert_permitted_types_to_required(permitted=(str, Path), required=Path, pos=0)
def save_images(path: Union[str, Path], images: np.ndarray, size_cap: float = 3.9) -> int:
    """
    Save a numpy array to a single .tif file. If size > 4GB then saved as a series of files. If path is not a file and
    already exists the default filename will be *images*.

    :param path: filename or absolute path
    :param images: numpy array (frames, y pixels, x pixels)
    :param size_cap: maximum size per file
    :returns: returns 0 if successful
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
    frames_per_file = calculate_frames_per_file(*images.shape[1:], size_cap=size_cap)
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
    except (RuntimeError, StopIteration):
        pass
    return 0


class _Metadata:
    def __init__(self, images: Optional[np.ndarray] = None):
        """
        Metadata object using for saving/loading binary images

        :param images: images in numpy array (frames, y-pixels, x-pixels)
        """
        self.frames, self.y, self.x, self.dtype = None, None, None, None

        if images is not None:
            self.frames, self.y, self.x = images.shape
            self.dtype = str(images.dtype)

    @classmethod
    def decode(cls: _Metadata, filename: Path) -> _Metadata:

        with open(str(filename), "r+") as file:
            attrs = load(file)

        metadata = _Metadata()
        for key, value in attrs.items():
            setattr(metadata, key, value)

        if "dtype" not in attrs:
            raise AttributeError("dtype not found")

        missing_keys = []
        for key in ["frames", "y", "x"]:
            if key not in attrs:
                missing_keys.append(key)

        if len(missing_keys) > 1:
            raise AttributeError("Require at least 2/3 shape keys")

        if len(missing_keys) == 1:
            setattr(metadata, missing_keys[0], -1)
            # set to negative one to let numpy figure it out

        return metadata

    def encode(self, filename: Path) -> int:
        meta = vars(self)
        with open(str(filename), "w+") as file:
            dump(meta, file)
        return 0
