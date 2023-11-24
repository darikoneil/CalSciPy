import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("QtAgg")

from CalSciPy.optogenetics import Photostimulation
from CalSciPy.roi_tools import Suite2PHandler
from bruker.protocols.protocols import generate_galvo_point_list


# file locs
base_path = Path("C:\\Users\\YUSTE\\Desktop\\EM0566_SET")
s2p_file = base_path.joinpath("suite2p").joinpath("plane0")
src_trgs_file = base_path.joinpath("targets_file.npy")
gpl_file = base_path.joinpath("TRIALS_15_D_EM0566.gpl")
prot_file = base_path.joinpath("TRIALS_15_D_EM0566.xml")

# load photostim source (no targets)
photostim = Photostimulation.import_rois(Suite2PHandler, s2p_file)

# load target source
targ_src = np.load(src_trgs_file, allow_pickle=True).item()

# target parameters
delay = 5000.0
point_interval = 4900.0

# add source targets
for trial in range(len(targ_src)):
    name = f"Trial {trial}"
    photostim.add_photostimulation_group(targ_src.get(trial), delay=delay, point_interval=point_interval, name=name)


# load GPL
src_gpl = generate_galvo_point_list(photostim, targets_only=True, name="src_targets", z_offset=21.44)
