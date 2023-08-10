from __future__ import annotations
from typing import Tuple, Union, Sequence
from pathlib import Path
from collections import ChainMap
from numbers import Number

import numpy as np

from .roi_tools import ROIHandler, Suite2PHandler


class Photostimulation:
    """
    Photostimulation object that defines patterned photostimulation during an optogenetic experiment

    :ivar rois: dictionary containing a collection of ROI objects for potential photostimulation
    :type rois: dict
    :ivar reference_image: a reference image containing the provided ROIs.
    :type reference_image: numpy.ndarray
    :ivar sequence: the sequence of individual photostimulation events
    """
    def __init__(self,
                 rois: dict,
                 reference_image: np.ndarray = None,
                 ):
        """
        Photostimulation object that defines patterned photostimulation during an optogenetic experiment

        :param rois: dictionary containing a collection of ROI objects for potential photostimulation
        :param reference_image: a reference image containing the provided ROIs.
        """

        self.rois = rois
        self.reference_image = reference_image
        self.sequence = None

    def __str__(self):
        return f"Photostimulation experiment targeting {self.targets} neurons from {len(self.rois)} total " \
               f"ROIs within {self.reference_image.shape[0]} x {self.reference_image.shape[1]} reference image (x, y)"

    @property
    def targets(self) -> int:
        return 15

    @property
    def neurons(self) -> int:
        return len(self.rois)

    @staticmethod
    def __name__() -> str:
        return "Photostimulation"

    @classmethod
    def import_rois(cls: Photostimulation, handler: ROIHandler = Suite2PHandler, *args, **kwargs) -> Photostimulation:
        """
        Class method builds a photostimulation instance using ROI Handler

        :param handler: desired ROIHandler
        :return: photostimulation instance
        """

        rois, reference_image = handler.load(*args, **kwargs)

        return Photostimulation(rois, reference_image=reference_image)

    def __repr__(self):
        return "Photostimulation(" + "".join([f"{key}: {value} " for key, value in vars(self).items()]) + ")"


class Group:
    def __init__(self):
        """
        Photostimulation group object containing the index of rois to stimulate
        and relevant stimulation parameters

        :ivar order: a tuple containing the identity and stimulation order of the rois in this group
        :type order: Tuple[int]
        :ivar repetitions: an integer indicating the number of times to repeat the stimulation
        :type repetitions: int
        """
        self.order = None
        self.repetitions = 1
