import pytest
from tests.conftest import retrieve_roi
from copy import deepcopy
from math import ceil
from operator import ge, le
import numpy as np

from CalSciPy.roi_tools import ROI


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
                    y, x = np.where(getattr(test_roi, attr))
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
    Fixture for the helper class

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
