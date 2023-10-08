import numpy as np

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib import pyplot as plt
import seaborn as sns


plt.style.use("CalSciPy.main")


from CalSciPy.simulations import SimulatedResponse

time = np.arange(-5, 5, 0.01)

sr = SimulatedResponse(peak=0, width=1, amplitude=1, reliability=1, noise=0, jitter=(0, 0, 0))

response = sr.respond(time)

fig, ax = plt.subplots(1, 1)
ax.plot(time, response)
