import pytest
from .conftest import retrieve_suite2p
# noinspection PyProtectedMember
from CalSciPy._helpers import BlockPrinting

import numpy as np
from copy import deepcopy

from CalSciPy.roi_tools import Suite2PHandler
from CalSciPy.optogenetics import Photostimulation, StimulationGroup, StimulationSequence, randomize_targets


"""
Testing suite for optogenetics (generic)

"""


@pytest.fixture()
def suite2p_handler_folder(request, temp_path):
    """
    Fixture for the handler helper class

    """
    return temp_path.joinpath("suite2p").joinpath(request.param)


@pytest.mark.parametrize("suite2p_handler_folder",
                         [folder for folder in retrieve_suite2p()],
                         indirect=["suite2p_handler_folder"])
def test_optogenetics_handler_import(suite2p_handler_folder):
    _ = Photostimulation.load_rois(Suite2PHandler, suite2p_handler_folder)


class Stim:
    """
    Helper object for photostimulation tests

    """
    def __init__(self, rois, reference_image):
        self.rois = rois
        self.reference_image = reference_image
        self.photostimulation = Photostimulation(deepcopy(self.rois), deepcopy(self.reference_image))


@pytest.fixture(scope="function")
def stim(sample_roi_map, sample_reference_image):
    return Stim(sample_roi_map, sample_reference_image)


class TestStim:
    def test_standard_properties(self, stim):
        assert (stim.photostimulation.num_neurons == len(stim.rois))
        assert (stim.photostimulation.reference_shape == stim.reference_image.shape)

    def test_exception_mismatched_reference_shape(self, stim):
        roi = stim.photostimulation.rois[0]
        roi.reference_shape = (256, 256)
        stim.photostimulation.add_photostimulation_group([0, 1])
        _ = stim.photostimulation.reference_shape
        assert (stim.photostimulation.stim_sequence[0].num_targets == 2)
        _ = stim.photostimulation.stim_sequence[0].reference_shape

    def test_add_photostim(self, stim):
        stim.photostimulation.add_photostimulation_group(
            ordered_index=[10, 20]
        )

        assert (stim.photostimulation.num_groups == 1)
        assert (stim.photostimulation.num_targets == 2)
        assert (len(stim.photostimulation.stim_sequence) == 1)

    def test_stim_indexing(self, stim):
        stim.photostimulation.add_photostimulation_group(
            ordered_index=[9, 19, 29]
        )

        assert (stim.photostimulation.stimulated_neurons == {9, 19, 29})
        assert (stim.photostimulation.roi_to_target(19) == 1)
        assert (stim.photostimulation.target_to_roi(1) == 19)

        remapped = stim.photostimulation.remapped_sequence
        assert (remapped[0].ordered_index == [0, 1, 2])

    def test_printing(self, stim):

        stim.photostimulation.add_photostimulation_group(
            ordered_index=[9, 19, 29]
        )

        # full
        with BlockPrinting():
            print(stim.photostimulation)
            print(stim.photostimulation.__name__())
            print(stim.photostimulation.__repr__())

        # stim sequence
            print(stim.photostimulation.stim_sequence)
            print(stim.photostimulation.stim_sequence.__name__())
            print(stim.photostimulation.stim_sequence.__repr__())

        # stim group
            print(stim.photostimulation.stim_sequence[0])
            print(stim.photostimulation.stim_sequence[0].__name__())
            print(stim.photostimulation.stim_sequence[0].__repr__())
