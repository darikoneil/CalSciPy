import numpy as np
from pathlib import Path

from CalSciPy.optogenetics import Photostimulation
from CalSciPy.optics.holo import SLM
from CalSciPy.optogenetics import randomize_targets
from visualize_optogenetics import *


data_folder = Path("D:\\SD_C\\pre_session_one\\results\\suite2p\\plane0")


photostimulation = Photostimulation.import_rois(folder=str(data_folder))

targets = randomize_targets(np.arange(photostimulation.num_neurons),
                            targets_per_group=5,
                            num_groups=1,
                            trials=1)[0]

for idx, target in enumerate(targets):
    photostimulation.add_photostimulation_group(target, name=f"Stimulation {idx}")


group = photostimulation.stim_sequence[0]

view_spiral_targets(photostimulation)

slm = SLM(brand="meadowlark", hdmi=False)

from slmsuite.holography.algorithms import Hologram

hologram.optimize(method="GS", maxiter=100)

hologram.plot_farfield(title="Test")
