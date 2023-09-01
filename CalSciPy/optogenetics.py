from __future__ import annotations
from typing import Sequence, Tuple, Iterable, Set, Union, Dict, Optional
from collections import UserList
from itertools import chain
from copy import deepcopy

import numpy as np

from .roi_tools import ROIHandler, Suite2PHandler, ROI
from ._calculations import multiple_random_groups_without_replacement


"""
Object-oriented approach to designing patterned optogenetic experiments
"""


class Photostimulation:
    """
    An instance of this class defines the patterned photostimulation during an optogenetic experiment. It is a
    standardized class for outlining parameters and selecting targets, and can be used as an argument to other
    functions for generating visuals, importing to microscopes, and other features.

    """
    def __init__(self,
                 rois: Dict[int, ROI],
                 reference_image: np.ndarray = None,
                 ):
        """
        An instance of this class defines the patterned photostimulation during an optogenetic experiment. It is a
        standardized class for outlining parameters and selecting targets, and can be used as an argument to other
        functions for generating visuals, importing to microscopes, and other features.

        :param rois: Dictionary mapping integer keys to :class:`ROI <CalSciPy.roi_tools.ROI>`\'s for potential
            photostimulation.

        :type rois: :class:`Dict <typing.Dict>`\[:class:`int`, :class:`ROI <CalSciPy.roi_tools.ROI>`\]

        :param reference_image: Reference image containing the provided ROIs.

        """
        #: :class:`dict`\: Dictionary mapping integer keys to :class:`ROI <CalSciPy.roi_tools.ROI>`\s
        #: for potential photostimulation.
        self.rois = rois

        #: :class:`ndarray <numpy.ndarray>`\, default: ``None``\: np.ndarray: Reference image containing
        #: the provided :class:`ROI <CalSciPy.roi_tools.ROI>`\'s
        self.reference_image = reference_image

        #: :class:`StimulationSequence <CalSciPy.optogenetics.StimulationSequence>`\: The sequence
        #: of :class:`StimulationGroup <CalSciPy.optogenetics.StimulationGroup>`\'s for this experiment
        self.stim_sequence = StimulationSequence()

    def __str__(self):
        return f"Photostimulation experiment targeting {self.num_targets} neurons from {self.num_neurons} total " \
               f"ROIs within {self.reference_image.shape[0]} x {self.reference_image.shape[1]} reference image (x, y)"

    @property
    def num_groups(self) -> int:
        """
        :Getter: Number of :class:`StimulationGroup <CalSciPy.optogenetics.StimulationGroup>`\'s in the
            :class:`StimulationSequence <CalSciPy.optogenetics.StimulationSequence>`
        :Getter Type: :class:`int`
        :Setter: This property cannot be set
        """
        return len(self.stim_sequence)

    @property
    def num_neurons(self) -> int:
        """
        :Getter: The total number of :class:`ROI <CalSciPy.roi_tools.ROI>`\'s in the roi map
        :Getter Type: :class:`int`
        :Setter: This property cannot be set
        """
        return len(self.rois)

    @property
    def num_targets(self) -> int:
        """
        :Getter: The number of neurons photostimulated
        :Getter Type: :class:`int`
        :Setter: This property cannot be set
        """
        return len(self.stimulated_neurons)

    @property
    def remapped_sequence(self) -> int:
        """
        :Getter: :class:`StimulationGroup <CalSciPy.optogenetics.StimulationGroup>`\'s with indices
            remapped to only include photostimulated :class:`ROI <CalSciPy.roi_tools.ROI>`\'s
        :Getter Type: :class:`int`
        :Setter: This property cannot be set
        """
        # copy to ensure no mutation
        remapped_sequence = deepcopy(self.stim_sequence)
        for group in remapped_sequence:
            group.ordered_index = [self.roi_to_target(target) for target in self.stimulated_neurons]
        return remapped_sequence

    @property
    def _roi_to_target_map(self) -> dict:
        """
        :Getter: A map whose key-value pairs represent the index of an :class:`ROI <CalSciPy.roi_tools.ROI>`
            in the original roi map & their index in new roi map only containing photostimulated
            :class:`ROI <CalSciPy.roi_tools.ROI>`\s.
`       :Getter Type: :class:`dict`
        :Setter: This property cannot be set
        """
        return dict(zip(
            self.stimulated_neurons,
            range(self.num_targets)
        ))

    @property
    def reference_shape(self) -> Tuple[int, int]:
        """
        :Getter: Shape of reference image
        :Getter Type: :class:`Tuple <typing.Tuple>`\[:class:`int`\, :class:`int`\]
        :Setter: This property cannot be set
        """
        shapes = [roi.reference_shape for roi in self.rois.values()]
        try:
            assert (len(set(shapes)) <= 1)
            return shapes[0]
        except AssertionError:
            print("Inconsistent reference_shape detected, selecting largest dimension for each axis")
            x_shape = max({x for _, x in shapes})
            y_shape = min({y for y, _ in shapes})
            return y_shape, x_shape

    @property
    def stimulated_neurons(self) -> Set[int]:
        """
        :Getter: The :class:`ROI <CalSciPy.roi_tools.ROI`\'s stimulated in the stimulation stim_sequence
        :Getter Type: :class:`Set <typing.Set>`\[:class:`int`\]
        :Setter: This property cannot be set
        """
        # Return unique neurons in the ordered indices of each group in the stim_sequence
        if self.stim_sequence is not None:
            return set(chain.from_iterable([group.ordered_index for group in self.stim_sequence]))

    @property
    def _target_to_roi_map(self) -> dict:
        """
        :Getter: Map of :class:`ROI <CalSciPy.roi_tools.ROI>`\'s remapped to only include photostimulated
            :class:`ROI <CalSciPy.roi_tools.ROI>`\'s
        :Getter Type: :class:`dict`
        :Setter: This property cannot be set
        """
        return dict(enumerate([roi for index, roi in enumerate(self.rois) if index in self.stimulated_neurons]))

    @staticmethod
    def __name__() -> str:
        return "Photostimulation"

    @classmethod
    def import_rois(cls: Photostimulation, handler: ROIHandler = Suite2PHandler, *args, **kwargs) -> Photostimulation:
        """
        Class method that builds a :class:`Photostimulation <CalSciPy.optogenetics.Photostimulation>` instance
        using the specified :class:`ROIHandler <CalSciPy.roi_tools.ROIHandler>`\. Any additional arguments or keyword
        arguments are passed to the :class:`ROIHandler <CalSciPy.roi_tools.ROIHandler>`\.

        :param handler: desired ROIHandler
        :type handler: :class:`ROIHandler <CalSciPy.roi_tools.ROIHandler>`
        :returns: A photostimulation instance
        :rtype: :class:`Photostimulation <CalSciPy.optogenetics.Photostimulation>`
        """

        rois, reference_image = handler.load(*args, **kwargs)

        return Photostimulation(rois, reference_image=reference_image)

    def add_photostimulation_group(self,
                                   ordered_index: Sequence[int],
                                   delay: float = 0.0,
                                   repetitions: int = 1,
                                   point_interval: float = 0.12,
                                   name: Optional[str] = None,
                                   ) -> Photostimulation:
        """
        Method that creates a :class:`StimulationGroup <CalSciPy.optogenetics.StimulationGroup>` and appends it to this
        :class:`Photostimulation <CalSciPy.optogenetics.Photostimulation>` instance's
        :class:`StimulationSequence <CalSciPy.optogenetics.StimulationSequence>` (that is, the attribute
        :attr:`stim_sequence`). More information on the arguments can be found in the documentation of
        :class:`StimulationGroup <CalSciPy.optogenetics.StimulationGroup>`\.


        :param ordered_index: Identity and order of the ROIs in this group

        :param delay: Delay before stimulating group

        :param repetitions: Number of times to repeat the stimulation group

        :param point_interval: Duration between stimulating each target in the stim_sequence (ms)

        :param name: Name of the group

        :type name: :class:`Optional <typing.Optional>`\[:class:`str`\], default: ``None``
        """
        rois = [self.rois.get(roi) for roi in ordered_index]
        self.stim_sequence.append(StimulationGroup(rois, ordered_index, delay, repetitions, point_interval, name))

    def roi_to_target(self, roi_index: int) -> int:
        """
        Converts a zero-indexed roi index to a zero-indexed target index

        :param roi_index: Index of the roi in the roi map

        :returns: Index of the roi in a target index
        """
        return self._roi_to_target_map.get(roi_index)

    def target_to_roi(self, target_index: int) -> int:
        """
        Converts a zero-indexed target index to the zero-indexed roi index

        :param target_index: index of the target

        :returns: Corresponding index in the roi map
        """
        return self._target_to_roi_map.get(target_index)

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
        Group of targets to be stimulated. Contains the index of rois to stimulate
        and relevant stimulation parameters.

        """
        #: :class:`float`\: Delay before stimulating group
        self.delay = delay

        #: :class:`str`\: Name of the group
        self.name = name

        #: :class:`Sequence <typing.Sequence>`\[:class:`int`\]: Identity and
        #: stimulation order of the :class:`ROI <CalSciPy.roi_tools.ROI>`\'s
        self.ordered_index = ordered_index

        #: :class:`float`\: Duration between stimulating each target in the stim_sequence (ms)
        self.point_interval = point_interval

        #: :class:`int`\: Number of times to repeat the stimulation group
        self.repetitions = repetitions

        #: :class:`Sequence <typing.Sequence>`\[:class:`ROI <CalSciPy.roi_tolls.ROI>`\]: Reference copy
        #: injected from the ROIs in photostimulation
        self.rois = rois

    def __str__(self):
        return f'Photostimulation group "{self.name}" containing {len(self.ordered_index)} targets with a ' \
               f'{self.point_interval} ms inter-point interval repeated {self.repetitions} times.'

    @property
    def num_targets(self) -> int:
        """
        :Getter: Number of roi targets in group
        :Getter Type: :class:`int`
        :Setter: This property cannot be set
        """
        return len(self.rois)

    @property
    def reference_shape(self) -> Tuple[int, int]:
        """
        :Getter: Shape of reference image
        :Getter Type: :class:`Tuple <typing.Tuple>`\[:class:`int`\, :class:`int`\]
        :Setter: This property cannot be set
        """
        shapes = [roi.reference_shape for roi in self.rois]
        try:
            assert (len(set(shapes)) <= 1)
            return shapes[0]
        except AssertionError:
            print("Inconsistent reference_shape detected, selecting largest dimension for each axis")
            x_shape = max({x for _, x in shapes})
            y_shape = min({y for y, _ in shapes})
            return y_shape, x_shape

    @staticmethod
    def __name__() -> str:
        return "Photostimulation StimulationGroup"

    def __repr__(self):
        return "StimulationGroup(" + "".join([f"{key}: {value}, " for key, value in vars(self).items()]) + ")"


class StimulationSequence(UserList):
    """
    The sequence of :class:`StimulationGroup <CalSciPy.optogenetics.StimulationGroup>`\'s to be stimulated in the
    :class:`Photostimulation <CalSciPy.optogenetics.Photostimulation>` experiment.

    """
    def __init__(self,
                 groups: Sequence = None,
                 delay: float = 0.0,
                 repetitions: int = 1,
                 interval: float = 0.0):
        """
        The sequence of :class:`StimulationGroup <CalSciPy.optogenetics.StimulationGroup>`\'s to be stimulated in the
        :class:`Photostimulation <CalSciPy.optogenetics.Photostimulation>` experiment.

        :param groups: The :class:`StimulationGroup <CalSciPy.optogenetics.StimulationGroup>`\'s to be stimulated

        :param delay: The initial delay before starting the sequence

        :param repetitions: The number of times to repeat the sequence

        :param interval: The interval between repetitions

        """
        #: :class:`int`\: Number of repetitions
        self.repetitions = repetitions

        #: :class:`float`\: Interval between repetitions
        self.interval = interval

        #: :class:`float`\: Delay before beginning stimulation sequence
        self.delay = delay

        super().__init__(initlist=groups)

    def __str__(self):
        return f"Photostimulation stim_sequence containing {len(self)} groups repeated {self.repetitions} " \
               f"times with an inter-repetition interval of {self.interval}"

    @staticmethod
    def __name__() -> str:
        return "Photostimulation stim_sequence"

    def __repr__(self):
        return "StimulationSequence(" + "".join([f"{key}: {value}, " for key, value in vars(self).items()]) + ")"


def randomize_targets(potential_targets: Union[Iterable, np.ndarray],
                      targets_per_group: int = 1,
                      num_groups: int = 1,
                      spatial_bin_size: int = None,
                      trials: int = 1,
                      ) -> Tuple[Tuple[int]]:
    """
    Randomly select targets to stimulate

    :param potential_targets: The rois available to target

    :param targets_per_group: The number of neurons to include in each group

    :param num_groups: The number of groups per trial

    :param spatial_bin_size: Not implemented

    :param trials: The number of trials to

    """

    if spatial_bin_size is not None:
        raise NotImplementedError

    return tuple(
        [multiple_random_groups_without_replacement(potential_targets, targets_per_group, num_groups)
         for _ in range(trials)]
    )
