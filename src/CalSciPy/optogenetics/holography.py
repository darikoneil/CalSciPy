from __future__ import annotations

import numpy as np

from slmsuite.holography.algorithms import Hologram


def literal_mask(t, roi):
    rc = roi.rc
    for pt in range(rc.shape[0]):
        y, x = rc[pt, :]
        t[y, x] = 1
    return t


def generate_masks(sequence, slm):

    rois = [photostimulation.rois.get(value) for key, value in photostimulation._target_to_roi_map.items()]

    target_size = (512, 512)

    target = np.zeros(target_size)

    for roi in rois:
        target = literal_mask(target, roi)

    hologram = Hologram(target, slm_shape=(512, 512))

    zoombox = hologram.plot_farfield(source=hologram.target, cbar=True)

    hologram.optimize(method='GS', maxiter=21)

    # Look at the associated near- and far- fields
    hologram.plot_nearfield(cbar=True)
    hologram.plot_farfield(limits=zoombox, cbar=True, title='FF Amp')


