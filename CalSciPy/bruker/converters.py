from __future__ import annotations
from typing import Union
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm as tq

from ..io_tools import _save_single_tif, _verbose_load_single_tif
from .._validators import convert_permitted_types_to_required
from .._files import calculate_frames_per_file
from .._calculations import generate_blocks
from .parsers import determine_imaging_content, generate_bruker_naming_convention
from ._helpers import print_image_description


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


@convert_permitted_types_to_required(permitted=(str, Path), required=Path, pos=0)
def repackage_bruker_tifs(input_folder: Union[str, Path],
                          output_folder: Union[str, Path],
                          channel: int = 0,
                          plane: int = 0
                          ) -> None:
    """
    This function repackages a folder containing .tif files exported by Bruker's Prairieview software into a sequence
    of <4 GB .tif stacks. Note that parameters channel and plane are **zero-indexed**.

    :param input_folder: folder containing a sequence of single frame .tif files exported by Bruker's Prairieview
    :param output_folder: empty folder where .tif stacks will be saved
    :param channel: specify channel
    :param plane: specify plane
    """
    num_channels, num_planes, num_frames, y, x = determine_imaging_content(input_folder)
    print_image_description(num_channels, num_planes, num_frames, y, x)

    naming_convention = generate_bruker_naming_convention(channel, plane, num_channels, num_planes)

    files = list(input_folder.glob(naming_convention))

    pbar = tq(total=num_frames)
    pbar.set_description("Repackaging...")

    block_size = calculate_frames_per_file(y, x)

    if num_frames > block_size:

        block_buffer = 0
        frame_index = list(range(num_frames))
        blocks = generate_blocks(frame_index, block_size, block_buffer)
        stack_index = 0

        for block in blocks:
            images = []
            for frame in block:
                images.append(_verbose_load_single_tif(files[frame], pbar))

            filename = generate_padded_filename(output_folder, stack_index)
            _save_single_tif(filename, np.concatenate(images, axis=0))

            stack_index += 1
    else:
        images = [_verbose_load_single_tif(file, pbar) for file in files]
        images = np.concatenate(images, axis=0)
        _save_single_tif(output_folder.joinpath("images.tif"), images)

    pbar.close()

