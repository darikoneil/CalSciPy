from __future__ import annotations
from typing import Tuple, Iterable

import numpy as np
from pathlib import Path

from CalSciPy.optogenetics import Photostimulation

from CalSciPy.optogenetics.cgh import SLM, Hologram, generate_target_mask
from CalSciPy.optogenetics.target_selection import randomize_targets

from dev.experimental.visualize_optogenetics import *


# parameters
data_folder = Path(".\\tests\\testing_samples\\suite2p\\plane0")

# create experiment
photostimulation = Photostimulation.import_rois(folder=data_folder)

# create targets
targets = randomize_targets(np.arange(photostimulation.num_neurons),
                            neurons_per_target=4,
                            num_targets=1,
                            trials=1)[0]

# add targets
for idx, target in enumerate(targets):
    photostimulation.add_photostimulation_group(target, name=f"Stimulation {idx}")

group = photostimulation.sequence[0]

view_targets(photostimulation)

masks = generate_target_mask(group)

masks = np.sum(masks, axis=0)

hologram = Hologram(masks, slm_shape=(512, 512))

zoombox = hologram.plot_farfield(source=hologram.target, cbar=True)

# Run 5 iterations of GS.
hologram.optimize(method="WGS-Kim", maxiter=100)

# Look at the associated near- and far- fields
hologram.plot_nearfield(cbar=True)
hologram.plot_farfield(limits=zoombox, cbar=True, title='FF Amp')
