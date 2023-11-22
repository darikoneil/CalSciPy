from __future__ import annotations
from typing import Union, Optional
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm as tq

from ..io_tools import _load_single_tif, _verbose_load_single_tif
from .._validators import convert_permitted_types_to_required, validate_extension
from .factories import BrukerImageFactory
from .parsers import determine_imaging_content, generate_bruker_naming_convention
from ._helpers import calc_ch_pl_comb, print_image_description


@convert_permitted_types_to_required(permitted=(str, Path), required=Path, pos=0)
def load_bruker_tifs(folder: Union[str, Path],
                     channel: Optional[int] = None,
                     plane: Optional[int] = None,
                     verbose: bool = True
                     ) -> Tuple[np.ndarray, ...]:  # noqa: C901
    """
    This function loads a **tuple** of images collected and converted to .tif files by Bruker's Prairieview software.
    Each channel and plane combination is loaded to a separate numpy array. Identification of multiple channels / planes is dependent on :func:`determine_imaging_content`.
    Images are loaded as unsigned 16-bit (:class:`numpy.uint16`), though note that raw bruker files are
    natively could be 12 or 13-bit.

    :param folder: folder containing a sequence of single frame tiff files
    :param channel: specific channel/s to load from dataset (zero-indexed)
    :param plane: specific plane/s to load from dataset (zero-indexed)\
    :param verbose: whether to print progress to terminal
    :return: a named tuple of numpy arrays (channel x plane) (frames, y-pixels, x-pixels, :class:`numpy.uint16`)
     """
    num_channels, num_planes, num_frames, y, x = determine_imaging_content(folder)

    if verbose:
        print_image_description(num_channels, num_planes, num_frames, y, x)

    if channel is not None and plane is not None:
        assert (channel < num_channels)
        assert (plane < num_planes)
        comb = calc_ch_pl_comb(channel, plane, exact=True)
    elif channel is None and plane is not None:
        assert (plane < num_planes)
        comb = calc_ch_pl_comb(range(channel), plane, exact=True)
    elif channel is not None and plane is None:
        assert (channel < num_channels)
        comb = calc_ch_pl_comb(channel, range(plane), exact=True)
    else:
        comb = calc_ch_pl_comb(num_channels, num_planes, exact=False)

    factory = BrukerImageFactory.create(comb)

    return factory(*[_load_bruker_tif_stack(folder, channel_, plane_, num_channels, num_planes)
                     for channel_, plane_ in comb])


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


def _load_bruker_tif_stack(input_folder: Path,
                           channel: int,
                           plane: int,
                           num_channels: int,
                           num_planes: int,
                           verbose: bool = True
                           ) -> np.ndarray:
    """
    Implementation function to load a single bruker tif stack

    :param input_folder: folder containing tif stack
    :param channel: selected channel to load
    :param plane: selected plane to load
    :param num_channels: total number of channels
    :param num_planes: total number of planes
    :param verbose: whether to return progress
    :returns: loaded images
    """
    naming_convention = generate_bruker_naming_convention(channel, plane, num_channels, num_planes)
    files = list(input_folder.glob(naming_convention))

    if verbose:
        pbar = tq(total=len(files))
        pbar.set_description("".join(["Loading Channel ", str(channel), " Plane ", str(plane)]))
        images = [_verbose_load_single_tif(file, pbar) for file in files]
        pbar.close()
    else:
        images = [_load_single_tif(file) for file in files]

    return np.concatenate(images, axis=0)


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
