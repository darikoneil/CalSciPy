from __future__ import annotations
from typing import Sequence, Tuple
from collections import UserList
from itertools import chain
from copy import deepcopy

import numpy as np
from numpy.typing import NDArray

from ..roi_tools import ROIHandler, Suite2PHandler


"""
Object-oriented approach to designing optogenetic experiments
"""


class Photostimulation:
    """
    Photostimulation object that defines patterned photostimulation during an optogenetic experiment

    """
    def __init__(self,
                 rois: dict,
                 reference_image: np.ndarray = None,
                 ):
        """
        Photostimulation object that defines patterned photostimulation during an optogenetic experiment

        :param rois: dictionary containing a collection of ROI objects for potential photostimulation
        :param reference_image: a reference image containing the provided ROIs.
        :param sequence: the sequence of photostimulated groups for this experiment
        """
        #: dict: dictionary containing a collection of :class:`ROI` objects for potential photostimulation
        self.rois = rois
        #: np.ndarray: a reference image containing the provided :class:`ROI`'s
        self.reference_image = reference_image
        #: StimulationSequence: The sequence of :class:`StimulationGroup`'s for this experiment
        self.sequence = StimulationSequence()

    def __str__(self):
        return f"Photostimulation experiment targeting {self.num_targets} neurons from {self.num_neurons} total " \
               f"ROIs within {self.reference_image.shape[0]} x {self.reference_image.shape[1]} reference image (x, y)"

    @property
    def num_groups(self) -> int:
        """
        Number of :class:`StimulationGroup`'s in the :class:`StimulationSequence`

        """
        return len(self.sequence)

    @property
    def num_neurons(self) -> int:
        """
        The total number of :class:`ROI`'s in the roi map

        """
        return len(self.rois)

    @property
    def num_targets(self) -> int:
        """
        The number of neurons photostimulated

        """
        return len(self.stimulated_neurons)

    @property
    def remapped_sequence(self) -> int:
        """
        :class:`StimulationGroup`'s with indices remapped to only include photostimulated :class:`ROI`'s

        """
        # copy to ensure no mutation
        remapped_sequence = deepcopy(self.sequence)
        for group in remapped_sequence:
            group.ordered_index = [self.roi_to_target(target) for target in self.stimulated_neurons]
        return remapped_sequence

    @property
    def _roi_to_target_map(self) -> dict:
        """
        A map whose key-value pairs represent the index of an :class:`ROI: in the original roi map &
        their index in new roi map only containing photostimulated :class:`ROI`'s

        """
        return dict(zip(
            self.stimulated_neurons,
            range(self.num_targets)
        ))

    def roi_to_target(self, roi_index: int) -> int:
        return self._roi_to_target_map.get(roi_index)

    @property
    def reference_shape(self) -> Tuple[int, int]:
        """
        Shape of reference image

        """
        shapes = [roi.reference_shape for roi in self.rois.values()]
        try:
            assert(len(set(shapes)) <= 1)
            return shapes[0]
        except AssertionError:
            print("Inconsistent reference_shape detected, selecting largest dimension for each axis")
            x_shape = max({x for _, x in shapes})
            y_shape = min({y for y, _ in shapes})
            return y_shape, x_shape

    @property
    def stimulated_neurons(self) -> set:
        """
        The :class:`ROI`'s stimulated in the stimulation sequence

        """
        # Return unique neurons in the ordered indices of each group in the sequence
        if self.sequence is not None:
            return set(chain.from_iterable([group.ordered_index for group in self.sequence]))

    @property
    def _target_to_roi_map(self) -> dict:
        """
        Map of :class:`ROI`'s remapped to only include photostimulated :class:`ROI`'s

        """
        return dict(enumerate([roi for index, roi in enumerate(self.rois) if index in self.stimulated_neurons]))

    def target_to_roi(self, target_index: int) -> int:
        return self._target_to_roi_map.get(target_index)

    @staticmethod
    def __name__() -> str:
        return "Photostimulation"

    @classmethod
    def import_rois(cls: Photostimulation, handler: ROIHandler = Suite2PHandler, *args, **kwargs) -> Photostimulation:
        """
        Class method that builds a :class:`Photostimulation` instance using the specified :class:`ROIHandler`.
        Any additional arguments or keyword arguments are passed to the :class:`ROIHandler`

        :param handler: desired ROIHandler
        :return: A photostimulation instance
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
        """
        Method that creates a :class:`StimulationGroup` and appends it to this
        :class:`Photostimulation` instance's :class:`StimulationSequence`
        (that is, the attribute :attr:`sequence`). More information on the arguments
        can be found in the documentation of :class:`StimulationGroup`


        :param ordered_index: a sequence containing the identity and order of the ROIs in this group
        :param delay: delay before stimulating group
        :param repetitions: the number of times to repeat the stimulation group
        :param point_interval: the duration between stimulating each target in the sequence (ms)
        :param name: name of the group
        """
        rois = [self.rois.get(roi) for roi in ordered_index]
        self.sequence.append(StimulationGroup(rois, ordered_index, delay, repetitions, point_interval, name))

    def __repr__(self):
        return "Photostimulation(" + "".join([f"{key}: {value} " for key, value in vars(self).items()]) + ")"


class StimulationGroup:
    def __init__(self,
                 rois: Sequence,
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
        #: Sequence[int]: a sequence containing the identity and stimulation order of the :class:`ROI`'s
        self.ordered_index = ordered_index
        #: float: the duration between stimulating each target in the sequence (ms)
        self.point_interval = point_interval
        #: int: the number of times to repeat the stimulation group
        self.repetitions = repetitions
        #: Sequence[ROIs]: this is a reference copy injected from the ROIs in photostimulation
        self.rois = rois

    def __str__(self):
        return f'Photostimulation group "{self.name}" containing {len(self.ordered_index)} targets with a ' \
               f'{self.point_interval} ms inter-point interval repeated {self.repetitions} times.'

    @staticmethod
    def __name__() -> str:
        return "Photostimulation StimulationGroup"

    @property
    def num_targets(self) -> int:
        """
        Number of roi targets in group

        """
        return len(self.rois)

    @property
    def reference_shape(self) -> Tuple[int, int]:
        """
        Shape of reference image

        """
        shapes = [roi.reference_shape for roi in self.rois]
        try:
            assert(len(set(shapes)) <= 1)
            return shapes[0]
        except AssertionError:
            print("Inconsistent reference_shape detected, selecting largest dimension for each axis")
            x_shape = max({x for _, x in shapes})
            y_shape = min({y for y, _ in shapes})
            return y_shape, x_shape

    def __repr__(self):
        return "StimulationGroup(" + "".join([f"{key}: {value}, " for key, value in vars(self).items()]) + ")"


class StimulationSequence(UserList):
    """
    Sequence of :class:`StimulationGroup`'s to be stimulated in the :class:`Photostimulation` experiment

    """
    def __init__(self,
                 groups: Sequence = None,
                 delay: float = 0.0,
                 repetitions: int = 1,
                 interval: float = 0.0):
        """
        Sequence of :class:`StimulationGroup`'s to be stimulated in the :class:`Photostimulation` experiment

        :param groups: a sequence of StimulationGroups
        """
        #: int: number of repetitions
        self.repetitions = repetitions
        #: float: interval between repetitions
        self.interval = interval
        #: float: delay before beginning sequence
        self.delay = delay

        super().__init__(initlist=groups)

    def __str__(self):
        return f"Photostimulation sequence containing {len(self)} groups repeated {self.repetitions} " \
               f"times with an inter-repetition interval of {self.interval}"

    @staticmethod
    def __name__() -> str:
        return "Photostimulation sequence"

    def __repr__(self):
        return "StimulationSequence(" + "".join([f"{key}: {value}, " for key, value in vars(self).items()]) + ")"
