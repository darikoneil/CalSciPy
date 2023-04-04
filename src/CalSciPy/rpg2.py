from __future__ import annotations
from typing import Union
import numpy as np
from pathlib import Path
import cv2
from tqdm.auto import tqdm
from tqdm import tqdm as tq
from .bruker import determine_imaging_content, _pretty_print_image_description
from .misc import PatternMatching



def generate_bruker_naming_convention(channel: int = 0, plane: int = 0, num_channels: int = 1, num_planes: int = 1) \
        -> str:
    """
    Generates the expected bruker naming convention for images collected with an arbitrary number of cycles & channels

    This function expects that the naming convention is {experiment_name}_Cycle00000_Ch0_000000.ome.tiff
    where the channel is one-indexed. The 5-digit cycle id represents the frame if using multiplane imaging and
    the 6-digit tag represents the plane. Otherwise, the 5-digit tag is static and the 6-digit tag represents the frame.
    Channel and plane are *zero-indexed*.

    :param channel: channel to produce name for
    :type channel: int = 0
    :param plane: plane to produce name for
    :type plane: int = 0
    :param num_channels: number of channels
    :type num_channels: int = 1
    :param num_planes: number of planes
    :type num_planes: int = 1
    :return:
    """
    # if channel + 1 > num_channels:
    #    raise AssertionError("Channel selection exceeds the amount of identified channels")
    # if plane + 1 > num_planes:
    #    raise AssertionError("Plane selection exceeds the amount of identified planes")

    with PatternMatching([num_channels, num_planes]) as case:
        if case([1, 1]):
            return ".ome"
        elif case([2, 1]):
            return "".join(["Ch", str(channel + 1), ])
        elif case([1, 2]):
            return "".join(["00000", str(plane + 1), ".ome"])
        elif case

    return ""


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


