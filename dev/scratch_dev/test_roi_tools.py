from __future__ import annotations

import pytest
from CalSciPy.roi_tools import ROI, _ROIBase, ApproximateROI, ROIHandler, Sequence, calculate_mask, calculate_radius, \
    calculate_centroid, identify_vertices, _validate_pixels

import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib import pyplot as plt

from itertools import chain

file = Path(".//tests//testing_samples//variables//sample_rois.npy")


"""
# generate sample neurons
x = np.arange(32, 512, 64)
centroid = [(x[i], x[i]) for i in range(x.shape[0])]
for_radii = list(range(2, len(centroid) + 1, 2))
for_radii = [radii * 2 for radii in for_radii]
rev_radii = reversed(list(range(len(centroid) - 3, len(centroid) * 2 - 3, 2)))
rev_radii = [radii * 2 for radii in for_radii]
radii = list(chain(*zip(for_radii, rev_radii)))
reference_shape = (512, 512)
sample_rois = {}


for neuron in range(len(centroid)):
    y, x = calculate_mask(centroid[neuron], radii[neuron], reference_shape=reference_shape)
    sample_rois[neuron] = {"ypix": y,
                           "xpix": x,
                           "reference_shape": reference_shape,
                           "centroid": centroid[neuron],
                           "radius": radii[neuron]
                           }

np.save(file, sample_rois, allow_pickle=True)
"""


sample_rois = np.load(file, allow_pickle=True).item()

roi = sample_rois.get(7)

test_roi = ROI(roi.get("xpix"), roi.get("ypix"), reference_shape=roi.get("reference_shape"))

from matplotlib.patches import Polygon  # noqa: E402
fig, ax = plt.subplots(1,1)
ax.imshow(test_roi.mask)
ax.add_patch(Polygon(test_roi.xy_vert, fill=False, color="black"))
test_roi.approx_method = "bound"
ax.add_patch(Polygon(test_roi.approximation.xy_vert, fill=False, color="blue"))
test_roi.approx_method="unbound"
ax.add_patch(Polygon(test_roi.approximation.xy_vert, fill=False, color="red"))

literal = {}
bound = {}
unbound = {}


# important stuff
keys = ("centroid", "radius", "vertices", "x_pixels", "y_pixels")

for key in key:


