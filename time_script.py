import numpy as np

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib import pyplot as plt
import seaborn as sns


plt.style.use("CalSciPy.main")


from CalSciPy.simulations import SimulatedResponse

time = np.arange(0, 55, 1)

peaks = np.arange(5, 45, 1).tolist()

width = [3 for _ in range(len(peaks))]

neurons = len(peaks)

sim_pop = [SimulatedResponse(peak=peaks[neuron],
                             width=width[neuron]
                             ) for neuron in range(neurons)]

responses = np.vstack([sim_pop[neuron].respond(time) for neuron in range(neurons)])

fig, ax = plt.subplots(1, 1)
s = sns.heatmap(responses, ax=ax, cmap="Spectral_r")
