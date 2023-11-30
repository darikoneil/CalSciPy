from __future__ import annotations
from typing import Sequence, Any, Union
from pathlib import Path

import numpy as np

from .._validators import convert_permitted_types_to_required
from .roi_tools import ROI, ROIHandler


class Suite2PHandler(ROIHandler):
    @staticmethod
    def convert_one_roi(roi: Any, reference_shape: Sequence[int, int] = (512, 512)) -> ROI:
        """
        Generates :class:`ROI <CalSciPy.roi_tools.ROI>` from `suite2p <https://www.suite2p.org>`_
        stat array

        :param roi: Dictionary containing one suite2p roi and its parameters

        :type roi: :class:`Any <typing.Any>`

        :param reference_shape: Reference_shape of the reference image containing the roi

        :type reference_shape: :class:`Sequence <typing.Sequence>`\[:class:`int`\, :class:`int`\],
            default: ``(512, 512)``

        :returns: ROI instance for the roi

        :rtype: :class:`ROI <CalSciPy.roi_tools.ROI>`
        """
        xpix = roi.get("xpix")[~roi.get("overlap")]
        ypix = roi.get("ypix")[~roi.get("overlap")]

        # must correct ypix indexing
        # ypix = reference_shape[0] - ypix

        return ROI(pixels=xpix,
                   y_pixels=ypix,
                   reference_shape=reference_shape,
                   properties=roi
                   )

    @staticmethod
    @convert_permitted_types_to_required(permitted=(str, Path), required=Path, pos=0, key="folder")
    def from_file(folder: Union[str, Path], *args, **kwargs) -> Sequence[np.ndarray, dict]:  # noqa: U100
        """
        Loads stat and ops from file

        :param folder: Folder containing `suite2p <https://www.suite2p.org>`_ data. The folder must contain the
            associated *stat.npy* & *ops.npy* files, though it is recommended the folder also contain the *iscell.npy*
            file.

        :type folder: :class:`Union <typing.Union>`\[:class:`str`\, :class:`Path <pathlib.Path>`\]

        :returns: Stat and ops

        :rtype: :class:`Sequence <typing.Sequence>`\[:class:`ndarray <numpy.ndarray>`\, :class:`dict`\]
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

            # ensure we embed index of original roi index
            for roi_idx in range(stat.shape[0]):
                stat[roi_idx]["roi_idx"] = roi_idx

            # also embed an index of neuron index & prune to neuronal rois only
            neuron_index = np.where(iscell[:, 0] == 1)[0]
            stat = stat[neuron_index]
            for neuron_idx in range(stat.shape[0]):
                stat[neuron_idx]["neuron_idx"] = neuron_index

        ops = np.load(folder.joinpath("ops.npy"), allow_pickle=True).item()

        return stat, ops

    @staticmethod
    def generate_reference_image(data_structure: Any) -> np.ndarray:
        """
         Generates an appropriate reference image from `suite2p <https://www.suite2p.org>`_ ops dictionary

        :param data_structure: Ops dictionary

        :returns: Reference image
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
