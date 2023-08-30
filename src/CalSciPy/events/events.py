from __future__ import annotations
from typing import Iterable, Optional

import numpy as np


def generate_raster(event_frames: Iterable[Iterable[int]], total_frames: Optional[int] = None) -> np.ndarray:
    """
    Generate raster from an iterable of iterables containing the spike (event) times for each neuron

    :param event_frames: Iterable containing an iterable identifying the event frames for each neuron

    :param total_frames: total number of frames

    :returns: Raster of neurons x total frames
    """

    if not total_frames:  # if total frames not provided we estimate by finding the very last event
        total_frames = np.max([event for events in event_frames for event in events]) + 1  # + 1 to account for 0-index
    _neurons = len(event_frames)
    event_matrix = np.full((_neurons, total_frames), 0, dtype=np.int32)

    # Merge Here - could be done more efficiently but not priority
    for _neuron in range(_neurons):
        for _event in event_frames[_neuron]:
            event_matrix[_neuron, _event] = 1
    return event_matrix
