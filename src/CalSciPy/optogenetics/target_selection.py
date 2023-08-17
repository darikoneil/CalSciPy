from __future__ import annotations
from typing import Tuple, Iterable, Union

import numpy as np


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
        [multiple_group_without_replacement(target_vector, neurons_per_target, num_targets) for _ in range(trials)]
    )


def multiple_group_without_replacement(sample_population, group_size, num_groups) -> Tuple:

    sample_population = np.asarray(sample_population)

    selections = []

    for group in range(num_groups):
        selections.append(np.random.choice(sample_population, size=group_size, replace=False).tolist())
        sample_population = np.setdiff1d(sample_population, np.hstack(selections))

    return tuple([tuple(target) for target in selections])
