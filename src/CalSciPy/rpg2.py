from __future__ import annotations
from typing import Union
from operator import eq
import numpy as np
from pathlib import Path
import cv2
from tqdm.auto import tqdm
from tqdm import tqdm as tq
from .bruker import determine_imaging_content, _pretty_print_image_description
from .misc import PatternMatching


def rpg_22(input_folder: Union[str, pathlib.Path], output_folder: Union[str, pathlib.Path],
           channel: int = 0, plane: int = 0) -> None:  # noqa: C901
    """
    Repackages a folder containing .tif files exported by Bruker's Prairieview software into a sequence of <4 GB .tif
    stacks. Channels are planes are **zero-indexed**.

    :param input_folder: folder containing a sequence of single frame .tif files exported by Bruker's Prairieview
    :type input_folder: str or pathlib.Path
    :param output_folder: empty folder where .tif stacks will be saved
    :type output_folder: str or pathlib.Path
    :param channel: optional input specifying channel
    :type channel: int = 0
    :param plane: optional input specifying plane
    :type plane: int = 0
    :rtype: None
    """
    files = list(input_folder.glob("*.tif"))
    num_channels, num_planes, num_frames, y, x = determine_imaging_content(input_folder)
    _pretty_print_image_description(num_channels, num_planes, num_frames, y, x)

    naming_convention = generate_bruker_naming_convention(channel, plane, num_channels, num_planes)


