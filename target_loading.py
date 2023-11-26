import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("QtAgg")
from matplotlib import pyplot as plt
import seaborn as sns

from CalSciPy.optogenetics import Photostimulation
from CalSciPy.roi_tools import Suite2PHandler
from CalSciPy.bruker.protocols import generate_galvo_point_list, generate_marked_points_protocol
from CalSciPy.bruker.meta import load_galvo_point_list, load_mark_points, load_saved_mark_points
from CalSciPy.bruker.data import load_voltage_recording
from CalSciPy.bruker.helpers import align_data, extract_frame_times


# file locs
# noinspection PyDictCreation
file_locs = {
    "base": Path("C:\\Users\\Darik\\Desktop\\EM0566_SET"),
}
file_locs["s2p"] = file_locs.get("base").joinpath("suite2p").joinpath("plane0")
file_locs["target"] = file_locs.get("base").joinpath("targets_file.npy")
file_locs["gpl"] = file_locs.get("base").joinpath("TRIALS_15_D_EM0566.gpl")
file_locs["smps"] = file_locs.get("base").joinpath("TRIALS_15_D_EM0566.xml")
file_locs["mps"] = file_locs.get("base").joinpath("EM0566_PRE_SCREEN_B_11_5_23-002_Cycle00001_MarkPoints.xml")
file_locs["t"] = file_locs.get("base").joinpath("EM0566_PRE_SCREEN_B_11_5_23-002.xml")
file_locs["vrf"] = file_locs.get("base").joinpath("EM0566_PRE_SCREEN_B_11_5_23-002_Cycle00001_VoltageRecording_001.xml")
file_locs["vrd"] = file_locs.get("base").joinpath("EM0566_PRE_SCREEN_B_11_5_23-002_Cycle00001_VoltageRecording_001.csv")
file_locs["env"] = file_locs.get("base").joinpath("EM0566_PRE_SCREEN_B_11_5_23-002.env")


# load photostim source (no targets)
photostim = Photostimulation.import_rois(Suite2PHandler, file_locs.get("s2p"))

source_info = {
    # load target source
    "targets": np.load(file_locs.get("target"), allow_pickle=True).item(),
    # target parameters
    "delay": 5000.0,
    "point_interval": 4900.0,
    "source_gpl": None,
    "source_smps": None,
}

# add source targets
for trial in range(len(source_info.get("targets"))):
    photostim.add_photostimulation_group(source_info.get("targets").get(trial),
                                         delay=source_info.get("delay"),
                                         point_interval=source_info.get("pointer_interval"),
                                         name=f"Trial {trial}")

# load src GPL
source_info["source_gpl"] = generate_galvo_point_list(photostim, targets_only=True, name="src_targets", z_offset=21.44)

source_info["source_smps"] = generate_marked_points_protocol(photostim,
                                                             targets_only=True,
                                                             file_path=str(file_locs.get("base")),
                                                             name="TRIALS_15_E_EM0566",
                                                             z_offset=21.44)
protocol_info = {
    "gpl": load_galvo_point_list(file_locs.get("gpl")),
    "smps": load_saved_mark_points(file_locs.get("smps")),
}


mps = load_mark_points(file_locs.get("mps"))

vrd = load_voltage_recording(file_locs.get("vrd"))

frame_times = extract_frame_times(file_locs.get("t"))

data = align_data(vrd, frame_times, fill=True)

idx = [gpe.name.split(" ")[-1] for gpe in protocol_info.get("gpl").galvo_points]

