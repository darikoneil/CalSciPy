from __future__ import annotations
from typing import Tuple, Iterable, Union
from .opto_objects import StimulationSequence

import numpy as np

from .._calculations import multiple_random_groups_without_replacement


def randomize_targets(target_vector: Union[Iterable, np.ndarray],
                      neurons_per_target: int = 1,
                      num_targets: int = 1,
                      spatial_bin_size: int = None,
                      trials: int = 1,
                      ) -> Tuple[Tuple[int]]:
    """
    Randomly select targets to stimulate

    :param target_vector:
    :param neurons_per_target:
    :param num_targets:
    :param spatial_bin_size:
    :param trials:
    :return:
    """
    return tuple(
        [multiple_random_groups_without_replacement(target_vector, neurons_per_target, num_targets)
         for _ in range(trials)]
    )
