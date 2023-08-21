from __future__ import annotations

import pytest
from CalSciPy.roi_tools import *

import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib import pyplot as plt

from itertools import chain

file = Path(".//tests//testing_samples//variables//sample_roi.npy")


"""
# generate sample neurons
x = np.arange(32, 512, 64)
centroid = [(x[i], x[i]) for i in range(x.shape[0])]
for_radii = list(range(2, len(centroid) + 1, 2))
rev_radii = reversed(list(range(len(centroid) - 3, len(centroid) * 2 - 3, 2)))
radii = list(chain(*zip(for_radii, rev_radii)))
reference_shape = (512, 512)
sample_rois = {}


for neuron in range(len(centroid)):
    y, x = calculate_mask(centroid[neuron], radii[neuron], shape=reference_shape)
    sample_rois[neuron] = {"ypix": y,
                           "xpix": x,
                           "reference_shape": reference_shape,
                           "centroid": centroid[neuron],
                           "radius": radii[neuron]
                           }

np.save(file, sample_rois, allow_pickle=True)
"""


sample_rois = np.load(file, allow_pickle=True).item()

test_rois = {}

for neuron in range(8):
    sample = sample_rois.get(neuron)
    test_rois[neuron] = ROI(sample.get("xpix"), sample.get("ypix"), sample.get("reference_shape"))


fig, ax = plt.subplots(1, 1)
ax.imshow(roi.mask)

radius = calculate_radius(roi.centroid, roi.rc_vert, method="all")
