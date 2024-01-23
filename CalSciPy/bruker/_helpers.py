from __future__ import annotations
from typing import Sequence, Union, Tuple
from itertools import product
from operator import eq

from .._backports import PatternMatching


def calc_ch_pl_comb(channels: Union[int, Sequence[int]],
                    planes: Union[int, Sequence[int]],
                    exact: bool = True
                    ) -> Tuple[Tuple[int, int], ...]:
    """
    Calculates all combinations of channels / planes that are requested. For ease of implementation, pass the exact flag
    to request combinations where there is only a single plane or channel, otherwise the integer will be considered the
    total number of channels or planes (e.g., if you request all planes at channel 2, use the exact flag. otherwise,
    the function will return all planes across two channels)

    :param channels:
    :param planes:
    :param exact:
    :returns: All combinations of channels x planes in a tuple of tuples
    """

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


def generate_bruker_naming_convention(channel: int,
                                      plane: int,
                                      num_channels: int = 1,
                                      num_planes: int = 1
                                      ) -> str:
    """
    Generates the expected bruker naming convention for images collected with an arbitrary number of cycles & channels

    This function expects that the naming convention is _Cycle00000_Ch0_000000.ome.tiff where the channel is
    one-indexed. The 5-digit cycle id represents the frame if using multiplane imaging and the 6-digit tag represents
    the plane. Otherwise, the 5-digit tag is static and the 6-digit tag represents the frame.

    Please note that the parameters channel and plane are *zero-indexed*.

    :param channel: channel to produce name for
    :param plane: plane to produce name for
    :param num_channels: number of channels
    :param num_planes: number of planes
    :returns: proper naming convention
    """
    if num_channels > 1:
        num_channels = 2
    if num_planes > 1:
        num_planes = 2
    with PatternMatching([num_channels, num_planes], [eq, eq]) as case:
        if case([1, 1]):
            return "*.ome.tif"
        elif case([2, 1]):
            return "".join(["*Ch", str(channel + 1), "*"])
        elif case([1, 2]):
            return "".join(["*00000", str(plane + 1), ".ome.tif"])
        elif case([2, 2]):
            return "".join(["*Ch", str(channel + 1), "_00000", str(plane + 1), ".ome.tif"])


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
