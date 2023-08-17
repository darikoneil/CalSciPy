from __future__ import annotations
from typing import Tuple, Iterable

import numpy as np
from pathlib import Path

from CalSciPy.opto import Photostimulation
from CalSciPy.bruker.protocols import generate_marked_points_protocol


from scratch_dev.visualize_optogenetics import *


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


def multiple_group_without_replacement(sample_population, group_size, num_groups):

    sample_population = np.asarray(sample_population)

    selections = []

    for group in range(num_groups):
        selections.append(np.random.choice(sample_population, size=group_size, replace=False).tolist())
        sample_population = np.setdiff1d(sample_population, np.hstack(selections))

    return tuple([tuple(target) for target in selections])


test_number = 2

# parameters
protocol_folder = Path("Z:\\Uncaging_Photostimulation_Tests\\08_14_2023").joinpath(f"test_{test_number}")

data_folder = protocol_folder.joinpath("suite2p")

# create experiment
photostimulation = Photostimulation.import_rois(folder=data_folder)

targets = randomize_targets(np.arange(photostimulation.total_neurons),
                            num_targets=5,
                            trials=1)[0]

for idx, target in enumerate(targets):
    photostimulation.add_photostimulation_group(target, name=f"Stimulation {idx}")


view_spiral_targets(photostimulation)

parameters = {"uncaging_laser_power": 1000, "spiral_revolutions": 0.01, "duration": 250.0}

mpl, gpl = generate_marked_points_protocol(photostimulation,
                                           targets_only=False,
                                           parameters=parameters,
                                           file_path=protocol_folder,
                                           name="test_protocol",
                                           z_offset=19.96
                                           )
