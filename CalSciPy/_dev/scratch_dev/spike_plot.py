from pathlib import Path
import numpy as np
from CalSciPy.events.event_visuals import SpikePlot


file = Path("D:\\DEM2\\retrieval\\results\\cascade")

approx = np.load(file.joinpath("spike_approx.npy"), allow_pickle=True)
prob = np.load(file.joinpath("spike_prob.npy"), allow_pickle=True)
times = np.load(file.joinpath("spike_times.npy"), allow_pickle=True)
dfof = np.load(file.parents[0].joinpath("dfof.npy"), allow_pickle=True)

_ = SpikePlot(prob, times, dfof, 30)
