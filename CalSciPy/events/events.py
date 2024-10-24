from __future__ import annotations
from typing import Iterable, Optional, Sequence

import numpy as np


def generate_raster(events: Sequence[Iterable[int]], samples: Optional[int] = None) -> np.ndarray:
    """
    Generate raster the spike (event) times for each neuron

    :param events: Sequence of iterables that identify the samples containing an event

    :param samples: Total number of samples

    :returns: Raster of n neurons x m samples
    """

    if not samples:  # if total frames not provided we estimate by finding the very last event
        samples = np.max([event for events_ in events for event in events_]) + 1  # + 1 to account for 0-index
    _neurons = len(events)
    event_matrix = np.full((_neurons, samples), 0, dtype=np.int32)

    # Merge Here - could be done more efficiently but not priority
    for _neuron in range(_neurons):
        for _event in events[_neuron]:
            event_matrix[_neuron, _event] = 1
    return event_matrix
