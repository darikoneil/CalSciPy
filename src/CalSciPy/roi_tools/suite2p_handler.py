from __future__ import annotations
from typing import Sequence, Any
from pathlib import Path

import numpy as np

from .roi_tools import ROI, ROIHandler


class Suite2PHandler(ROIHandler):
    @staticmethod
    def convert_one_roi(roi: Any, reference_shape: Sequence[int, int] = (512, 512)) -> ROI:
        """
        Generates ROI from suite2p stat array

        :param roi: dictionary containing one suite2p roi
        :param reference_shape: reference_shape of the reference image containing the roi
        :return: ROI instance for the roi
        """
        xpix = roi.get("xpix")[~roi.get("overlap")]
        ypix = roi.get("ypix")[~roi.get("overlap")]
        return ROI(pixels=xpix,
                   y_pixels=ypix,
                   reference_shape=reference_shape,
                   properties=roi
                   )

    @staticmethod
    def from_file(folder: Path, *args, **kwargs) -> Sequence[np.ndarray, dict]:  # noqa: U100
        """

        :param folder: folder containing suite2p data. The folder must contain the associated "stat.npy"
            & "ops.npy" files, though it is recommended the folder also contain the "iscell.npy" file.

        :returns: "stat" and "ops"
        """

        # append suite2p + plane if necessary
        if "suite2p" not in str(folder):
            folder = folder.joinpath("suite2p")

        if "plane" not in str(folder):
            folder = folder.joinpath("plane0")

        stat = np.load(folder.joinpath("stat.npy"), allow_pickle=True)

        # use only neuronal rois if iscell is provided
        try:
            iscell = np.load(folder.joinpath("iscell.npy"), allow_pickle=True)
        except FileNotFoundError:
            stat[:] = stat
        else:
            stat = stat[np.where(iscell[:, 0] == 1)[0]]

        ops = np.load(folder.joinpath("ops.npy"), allow_pickle=True).item()

        return stat, ops

    @staticmethod
    def generate_reference_image(data_structure: Any) -> np.ndarray:
        """
         Generates an appropriate reference image from suite2p ops dictionary

        :param data_structure: ops dictionary
        :return: reference image
        """

        true_shape = (data_structure.get("Ly"), data_structure.get("Lx"))

        # Load Vcorr as our reference image
        try:
            reference_image = data_structure.get("Vcorr")
            assert (reference_image is not None)
        except (KeyError, AssertionError):
            reference_image = np.ones(true_shape)

        # If motion correction cropped Vcorr, append minimum around edges
        if reference_image.shape != true_shape:
            true_reference_image = np.ones(true_shape) * np.min(reference_image)
            x_range = data_structure.get("xrange")
            y_range = data_structure.get("yrange")
            true_reference_image[y_range[0]: y_range[-1], x_range[0]:x_range[-1]] = reference_image
            return true_reference_image
        else:
            return reference_image
