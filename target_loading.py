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
from CalSciPy.bruker.xml.xmlobj import GalvoPoint
from CalSciPy.conversion import interpolate_traces
from CalSciPy.color_scheme import COLORS
from CalSciPy.bruker.data import load_bruker_tifs
from CalSciPy.io_tools import load_binary

# file locs
# noinspection PyDictCreation
file_locs = {
    "base": Path("C:\\Users\\YUSTE\\Desktop\\EM0566_SET"),
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
file_locs["traces"] = file_locs.get("base").joinpath("traces.npy")
file_locs["iscell"] = file_locs.get("base").joinpath("iscell.npy")
file_locs["bruker_tifs"] = Path("D:\\EM0566\\stimulation_session_one\\imaging")
file_locs["images"] = Path("D:\\EM0566\\stimulation_session_one\\results")

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

data = align_data(vrd, frame_times, fill=False)

neuron_idx = [gpe.name.split(" ")[-1] for gpe in protocol_info.get("gpl").galvo_points if isinstance(gpe, GalvoPoint)]

#
pre_frames = 2000
post_frames = 2000
stim_threshold = 4.00
stim = data.STIM.to_numpy().copy()
stim_idx = np.where(stim >= stim_threshold)[0]
stims = np.where(np.diff(stim_idx) > 1)[0]
stims += 1
stims = np.insert(stims, 0, 0, axis=0)
stim_trials = [range(stim_idx[stim] - pre_frames, stim_idx[stim] + post_frames) for stim in stims.tolist()]

#
traces = np.load(file_locs.get("traces"), allow_pickle=True)
# iscell = np.load(file_locs.get("iscell"), allow_pickle=True)
# neurons = np.where(iscell[:, 0] == 1)[0]
# n_traces = traces[neurons, :]
#
int_traces = interpolate_traces(traces, data, "nearest").to_numpy().T
#
trial_traces = [int_traces[:, idx] for idx in stim_trials]

trial_traces = np.dstack(trial_traces).swapaxes(0, 2).swapaxes(1, 2)
#

responses = np.sum(trial_traces, axis=0).sum(axis=0)

time = np.arange(-2, 2, 1/1000)

responses_delta = np.insert(np.diff(responses), 0, 0, axis=0)

images = load_binary(file_locs.get("images"), mapped=True, mode="r")
images = np.sum(images, axis=1).sum(axis=1)
int_images = interpolate_traces(np.reshape(images, (1, -1)), data, method="nearest").to_numpy().T
trial_images = [int_images[:, idx] for idx in stim_trials]
trial_images = np.dstack(trial_images).swapaxes(0, 2).swapaxes(1, 2)
images_responses = np.sum(trial_images, axis=0).flatten()
images_deriv = np.insert(np.diff(images_responses).T, 0, 0, axis=0)

brukers = load_bruker_tifs(file_locs.get("bruker_tifs"))[0]
brukers = np.sum(brukers, axis=1).sum(axis=1)
int_brukers = interpolate_traces(np.reshape(brukers, (1, -1)), data, method="nearest").to_numpy().T
trial_brukers = [int_brukers[:, idx] for idx in stim_trials]
trial_brukers = np.dstack(trial_brukers).swapaxes(0, 2).swapaxes(1, 2)
bruker_responses = np.sum(trial_brukers, axis=0).flatten()
bruker_deriv = np.insert(np.diff(bruker_responses), 0, 0, axis=0)

plot_idx = 0

plot_opts = (responses, responses_delta, images_responses, images_deriv, bruker_responses, bruker_deriv)
plot_var = plot_opts[plot_idx]
plt.style.use("CalSciPy.main")

fig, ax = plt.subplots(1, 1)
ax.plot(time, plot_var, lw=3, color=COLORS.BLUE)
ax.vlines(0, np.min(plot_var), np.max(plot_var), lw=1, color=COLORS.BLACK, ls="--")
ax.margins(0)
ax.set_xlabel("Time (s)")
