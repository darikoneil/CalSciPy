from CalSciPy.optics.holo import SLM
from CalSciPy.roi_tools import calculate_mask
from slmsuite.holography.algorithms import Hologram
import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib import pyplot as plt


slm_size = (512, 512)

targets = [calculate_mask(centroid=(target[0], target[1]), radii=10, reference_shape=(512, 512)) for target in
           zip((128, 255, 383), (128, 255, 255))]

target = np.sum(targets, axis=0)

fig, ax = plt.subplots(1, 1)
ax.imshow(target)

# slm = SLM(brand="meadowlark")

hologram = Hologram(target, slm_shape=(512, 512))

hologram.optimize(method="GS", maxiter=20)

hologram.plot_nearfield(padded=False, cbar=True)

hologram.plot_farfield(cbar=True, units="knm")
