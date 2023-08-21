import pytest
from tests.conftest import retrieve_roi
from copy import deepcopy

import numpy as np

from CalSciPy.roi_tools import ROI


class SampleROI:
    """
    Helper class for roi tools tests for an example roi
    """
    def __init__(self, sample_roi):

        # record true inputs
        self.ypix = sample_roi.get("ypix")
        self.xpix = sample_roi.get("xpix")
        self.centroid = sample_roi.get("centroid")
        self.radius = sample_roi.get("radius")
        self.reference_shape = sample_roi.get("reference_shape")

        self.test_roi = ROI(xpix=self.xpix.copy(),
                            ypix=self.ypix.copy(),
                            reference_shape=deepcopy(self.reference_shape)
                            )

    def validate_attrs(self):
        for attr in vars(self.test_roi):
            if hasattr(self, attr):
                test_attr = getattr(self.test_roi, attr)
                true_attr = getattr(self, attr)

                if isinstance(test_attr, (float, np.ndarray)):
                    np.testing.assert_allclose(test_attr, true_attr, atol=0.75)
                else:
                    assert(getattr(self, attr) == getattr(self.test_roi, attr))


@pytest.fixture()
def roi_helper(request):
    """
    Fixture for the helper class

    """
    return SampleROI(request.param)


@pytest.mark.parametrize("roi_helper", [roi for roi in retrieve_roi()], indirect=["roi_helper"])
class TestROI:
    """
    Actual test class

    """
    def test_initial_attributes(self, roi_helper):
        roi_helper.validate_attrs()
