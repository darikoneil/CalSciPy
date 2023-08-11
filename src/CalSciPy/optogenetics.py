from __future__ import annotations
from typing import Tuple, Union, Sequence
from pathlib import Path
from collections import ChainMap, UserList
from itertools import chain
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
        self.sequence = Sequence()

    def __str__(self):
        return f"Photostimulation experiment targeting {self.targets} neurons from {self.total_neurons} total " \
               f"ROIs within {self.reference_image.shape[0]} x {self.reference_image.shape[1]} reference image (x, y)"

    @property
    def target_mapping(self) -> int:
        return dict(zip(
            self.stimulated_neurons,
            range(self.targets)
        ))

    @property
    def groups(self) -> int:
        return len(self.sequence)

    @property
    def stimulated_neurons(self) -> set:
        # Return unique neurons in the ordered indices of each group in the sequence
        if self.sequence is not None:
            return set(chain.from_iterable([group.ordered_index for group in self.sequence]))

    @property
    def targets(self) -> int:
        """
        The number of neurons photostimulated

        """
        return len(self.stimulated_neurons)

    @property
    def total_neurons(self) -> int:
        """
        The total number of neurons in the image

        """
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

    def add_photostimulation_group(self,
                                   ordered_index: Sequence[int],
                                   delay: float = 0.0,
                                   repetitions: int = 1,
                                   point_interval: float = 0.12,
                                   name: str = None,
                                   ) -> Photostimulation:

        self.sequence.append(Group(ordered_index, delay, repetitions, point_interval, name))

    def __repr__(self):
        return "Photostimulation(" + "".join([f"{key}: {value} " for key, value in vars(self).items()]) + ")"


class Group:
    def __init__(self,
                 ordered_index: Sequence[int],
                 delay: float = 0.0,
                 repetitions: int = 1,
                 point_interval: float = 0.12,
                 name: str = None,
                 ):
        """
        Photostimulation group object containing the index of rois to stimulate
        and relevant stimulation parameters

        """
        #: float: delay before stimulating group
        self.delay = delay
        #: str: name of the group
        self.name = name
        #: Sequence[int]: a sequence containing the identity and stimulation order of the rois in this group
        self.ordered_index = ordered_index
        #: float: the duration between stimulating each target in the sequence (ms)
        self.point_interval = point_interval
        #: int: the number of times to repeat the stimulation group
        self.repetitions = repetitions

    def __str__(self):
        return f'Photostimulation Group "{self.name}" containing {len(ordered_index)} targets with a ' \
               f'{self.point_interval} inter-point interval repeated {self.repetitions} times.'

    @staticmethod
    def __name__():
        return "Photostimulation Group"

    def __repr__(self):
        return "Group(" + "".join([f"{key}: {value}, " for key, value in vars(self).items()]) + ")"


class Sequence(UserList):
    """
    Photostimulation Sequence

    """
    def __init__(self,
                 groups: Sequence = None,
                 repetitions: int = 1,
                 interval: float = 0.0):

        #: int: number of repetitions
        self.repetitions = repetitions,
        #: float: interval between repetitions
        self.interval = interval

        super().__init__(initlist=groups)

    def __str__(self):
        return f"Photostimulation Sequence containing {len(self)} groups repeated {self.repetitions} " \
               f"times with an inter-repetition interval of {self.interval}"

    @staticmethod
    def __set_name__() -> str:
        return "Photostimulation Sequence"
