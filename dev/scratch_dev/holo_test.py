from __future__ import annotations
from typing import Tuple, Iterable

import numpy as np
from pathlib import Path

from CalSciPy.optogenetics import Photostimulation
from CalSciPy.bruker.protocols import generate_marked_points_protocol

from CalSciPy.optogenetics.cgh import SLM
from CalSciPy.optogenetics.target_selection import randomize_targets
from slmsuite.holography.algorithms import Hologram

from dev.experimental.visualize_optogenetics import *


# parameters
data_folder = Path(".\\tests\\testing_samples\\suite2p\\plane0")

# create experiment
photostimulation = Photostimulation.import_rois(folder=data_folder)

# create targets
targets = randomize_targets(np.arange(photostimulation.num_neurons),
                            neurons_per_target=5,
                            num_targets=1,
                            trials=1)[0]

for idx, target in enumerate(targets):
    photostimulation.add_photostimulation_group(target, name=f"Stimulation {idx}")

