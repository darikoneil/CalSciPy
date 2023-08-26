import pytest
from tests.conftest import retrieve_roi, retrieve_suite2p
from tests.helpers import BlockPrinting

from copy import deepcopy
from math import ceil
from operator import ge, le
from abc import abstractmethod

import numpy as np

# noinspection PyProtectedMember
from CalSciPy.roi_tools import ROI, Suite2PHandler, ROIHandler

# noinspection PyProtectedMember
from CalSciPy.roi_tools.roi_tools import ROIBase


"""
Testing suite for ROIs. Functions not attached to classes are naturally tested by class tests at the moment

"""


class SampleROI:
    """
    Helper class for roi tools tests for an example roi
    """
    def __init__(self, sample_roi):

        # record true inputs
        self.y_pixels = sample_roi.get("ypix")
        self.x_pixels = sample_roi.get("xpix")
        self.centroid = sample_roi.get("centroid")
        self.radius = sample_roi.get("radius")
        self.reference_shape = sample_roi.get("reference_shape")

        # record true approximations
        self.literal = sample_roi.get("literal")
        self.bound = sample_roi.get("bound")
        self.unbound = sample_roi.get("unbound")

    def generate_test_roi_single_rc_arg(self):
        pixels = np.vstack([self.y_pixels, self.x_pixels]).T
        return ROI(pixels=pixels,
                   reference_shape=deepcopy(self.reference_shape)
                   )

    def generate_test_roi_x_and_y_args(self):
        return ROI(pixels=self.x_pixels.copy(),
                   y_pixels=self.y_pixels.copy(),
                   reference_shape=deepcopy(self.reference_shape)
                   )

    def validate_approximations(self, comparison, approx_roi, test_roi):
        assert(comparison(approx_roi.radius, test_roi.radius))

    def validate_attrs(self, test_roi):
        for attr in vars(test_roi):
            if hasattr(self, attr):
                test_attr = getattr(test_roi, attr)
                true_attr = getattr(self, attr)

                if isinstance(test_attr, (float, np.ndarray)):

                    # true radius is an int but pixels are discrete. we might be slightly lower when calculating
                    # from image. therefore we take the larger of the two nearest integers
                    if attr == "radius":
                        test_attr = ceil(test_attr)

                    np.testing.assert_equal(test_attr, true_attr)

                else:
                    assert(getattr(self, attr) == getattr(test_roi, attr))
            else:
                # check mask
                if attr == "mask":
                    print("mask!")
                    y, x = np.where(test_roi.mask)
                    assert(np.testing.assert_equal(y, self.y_pixels))
                    assert(np.testing.assert_equal(x, self.x_pixels))

    def validate_exceptions(self):
        # plane is not yet implemented
        with pytest.raises(NotImplementedError):
            ROI(self.x_pixels, self.y_pixels, plane=0)
        # z_pixels is not yet implemented
        with pytest.raises(NotImplementedError):
            ROI(self.x_pixels, self.y_pixels, z_pixels=np.ones_like(self.x_pixels))
        # approximation ellipse not yet implemented
        with pytest.raises(NotImplementedError):
            ROI(self.x_pixels, self.y_pixels, method="ellipse")


@pytest.fixture()
def sample_roi(request):
    """
    Fixture for the roi helper class

    """
    return SampleROI(request.param)


@pytest.mark.parametrize("sample_roi", [roi for roi in retrieve_roi()], indirect=["sample_roi"])
class TestROI:
    """
    Actual test class

    """
    def test_x_y_arg(self, sample_roi):
        test_roi = sample_roi.generate_test_roi_x_and_y_args()
        sample_roi.validate_attrs(test_roi)

    def test_rc_arg(self, sample_roi):
        test_roi = sample_roi.generate_test_roi_single_rc_arg()
        sample_roi.validate_attrs(test_roi)

    def test_literal(self, sample_roi):
        test_roi = sample_roi.generate_test_roi_x_and_y_args()
        test_roi.approx_method = "literal"
        literal_roi = test_roi.approximation
        sample_roi.validate_attrs(literal_roi)

    def test_bound(self, sample_roi):
        test_roi = sample_roi.generate_test_roi_x_and_y_args()
        test_roi.approx_method = "bound"
        sample_roi.validate_approximations(le, test_roi.approximation, test_roi)

    def test_unbound(self, sample_roi):
        test_roi = sample_roi.generate_test_roi_x_and_y_args()
        test_roi.approx_method = "unbound"
        sample_roi.validate_approximations(ge, test_roi.approximation, test_roi)

    def test_exceptions(self, sample_roi):
        sample_roi.validate_exceptions()

    def test_mask(self, sample_roi):
        test_roi = sample_roi.generate_test_roi_x_and_y_args()
        _ = test_roi.mask  # why am I not being registered as called during tests?

    def test_magic(self, sample_roi):
        with BlockPrinting():
            test_roi = sample_roi.generate_test_roi_x_and_y_args()
            print(test_roi.__str__())
            print(test_roi.__repr__())
            print(test_roi.__name__())
            print(test_roi.approximation.__str__())
            print(test_roi.approximation.__repr__())
            print(test_roi.approximation.__name__())

    def test_abstract(self, sample_roi):
        with pytest.raises(TypeError):
            roi_base = ROIBase(pixels=sample_roi.x_pixels, y_pixels=sample_roi.y_pixels)


class Handler:

    @abstractmethod
    def from_file(self, handler_folder):
        ...

    @abstractmethod
    def test_from_file(self, handler_folder):
        ...

    @abstractmethod
    def test_reference_image(self, handler_folder):
        ...

    @abstractmethod
    def test_conversion(self, handler_folder):
        ...

    @abstractmethod
    def test_import_rois(self, handler_folder):
        ...

    @abstractmethod
    def test_load(self, handler_folder):
        ...


@pytest.fixture()
def handler_folder(request, temp_path):
    """
    Fixture for the handler helper class

    """
    return temp_path.joinpath("suite2p").joinpath(request.param)


@pytest.mark.parametrize("handler_folder", [folder for folder in retrieve_suite2p()], indirect=["handler_folder"])
class TestSuite2PHandler:

    def from_file(self, handler_folder):
        return Suite2PHandler.from_file(handler_folder)

    def test_from_file(self, handler_folder):
        assert len(self.from_file(handler_folder)) == 2

    def test_reference_image(self, handler_folder):
        _, ops = self.from_file(handler_folder)
        _ = Suite2PHandler.generate_reference_image(ops)

    def test_conversion(self, handler_folder):
        stat, _ = self.from_file(handler_folder)
        roi = stat[0]
        _ = Suite2PHandler.convert_one_roi(roi)

    def test_import_rois(self, handler_folder):
        stat, ops = self.from_file(handler_folder)
        reference_shape = Suite2PHandler.generate_reference_image(ops).shape
        _ = Suite2PHandler.import_rois(stat, reference_shape)

    def test_load(self, handler_folder):
        _ = Suite2PHandler.load(handler_folder)
