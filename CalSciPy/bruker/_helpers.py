from __future__ import annotations
from typing import Sequence, Union, Tuple
from pathlib import Path
from itertools import product
from xml.etree import ElementTree

import numpy as np
import pandas as pd

from .._validators import convert_permitted_types_to_required


def calc_ch_pl_comb(channels: Union[int, Sequence[int]],
                    planes: Union[int, Sequence[int]],
                    exact: bool = True
                    ) -> Tuple[Tuple[int, int], ...]:

    if not isinstance(channels, Sequence):
        if exact:
            channels = [channels, ]
        else:
            channels = range(channels)

    if not isinstance(planes, Sequence):
        if exact:
            planes = [planes, ]
        else:
            planes = range(planes)

    return tuple(product(channels, planes))


@convert_permitted_types_to_required(permitted=(str, Path), required=str, pos=0)
def extract_frame_times(filename: Union[str, Path]) -> pd.DataFrame:
    """
    Function to extract the relative frame times from a PrairieView imaging session's primary .xml file

    :param: filename
    :returns: dataframe containing time (index, ms) x imaging frame (*zero-indexed*)
    """
    tree = ElementTree.parse(filename)
    root = tree.getroot()

    # assert expected
    # child_tags = [child.tag for child in root]
    # expected_tags = ("SystemIDs", "PVStateShard", "Sequence")

    # for tag_ in expected_tags:
    #   assert (tag_ in child_tags), "XML follows unexpected structure"

    # Since expected, let's grab frame sequence
    sequence = root.find("Sequence")
    # use set comprehension to avoid duplicates
    relative_frame_times = {frame.attrib.get("relativeTime") for frame in sequence if "relativeTime" in frame.attrib}
    # convert to float (appropriate type) & sort chronologically
    relative_frame_times = sorted([float(frame) for frame in relative_frame_times])
    frames = range(len(relative_frame_times))
    # convert to same type as analog data to avoid people getting gotcha'd by pandas
    frames = np.array(frames).astype(np.float64)
    # convert to milliseconds, create new array to avoid people getting gotcha'd by pandas
    frame_times = np.array(relative_frame_times) * 1000
    # round to each millisecond
    frame_times = np.round(frame_times).astype(np.int64)
    # make index
    frame_times = pd.Index(data=frame_times, name="Time (ms)")
    # make dataframe
    return pd.DataFrame(data=frames, index=frame_times, columns=["Imaging Frame"])


def print_image_description(channels: int,
                            planes: int,
                            frames: int,
                            height: int,
                            width: int
                            ) -> None:
    """
    Function prints the description of an imaging dataset.

    :param channels: number of channels
    :param planes: number of planes
    :param frames: number of frames
    :param height: y-pixels
    :param width: x-pixels
    """
    msg = f"\nImages: {channels * planes * frames}"
    msg += f"\nChannels\t\t{channels}"
    msg += f"\nPlanes\t\t{planes}"
    msg += f"\nFrames\t\t{frames}"
    msg += f"\nHeight\t\t{height}"
    msg += f"\nWidth\t\t{width}"
    msg += "\n"

    print(msg)
