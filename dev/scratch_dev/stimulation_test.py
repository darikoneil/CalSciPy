from __future__ import annotations
from typing import Tuple, Iterable

import numpy as np
from pathlib import Path

from CalSciPy.optogenetics import Photostimulation
from CalSciPy.bruker.protocols import generate_marked_points_protocol

from CalSciPy.optogenetics.cgh import SLM
from slmsuite.holography.algorithms import Hologram

from dev.experimental.visualize_optogenetics import *


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
protocol_folder = Path(".\\tests\\testing_samples\\suite2p\\plane0")

data_folder = protocol_folder

# create experiment
photostimulation = Photostimulation.import_rois(folder=data_folder)

targets = randomize_targets(np.arange(photostimulation.num_neurons),
                            num_targets=5,
                            trials=1)[0]

for idx, target in enumerate(targets):
    photostimulation.add_photostimulation_group(target, name=f"Stimulation {idx}")


view_spiral_targets(photostimulation)


def literal_mask(t, roi):
    rc = roi.rc
    for pt in range(rc.shape[0]):
        y, x = rc[pt, :]
        t[y, x] = 1
    return t


rois = [photostimulation.rois.get(value) for key, value in photostimulation.target_to_roi.items()]

# rois = [photostimulation.rois.get(22)]

target_size = (512, 512)

target = np.zeros(target_size)

for roi in rois:
    target = literal_mask(target, roi)

hologram = Hologram(target, slm_shape=(512, 512))

zoombox = hologram.plot_farfield(source=hologram.target, cbar=True)

# Run 5 iterations of GS.
hologram.optimize(method='GS', maxiter=5)

# Look at the associated near- and far- fields
hologram.plot_nearfield(cbar=True)
hologram.plot_farfield(limits=zoombox, cbar=True, title='FF Amp')

