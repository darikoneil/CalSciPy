import pytest

import numpy as np

from CalSciPy.images import gaussian_filter, median_filter
"""
Test suite for image processing filters

"""


@pytest.mark.parametrize("method", ("gaussian", "median"))
def test_image_filters(sample_images, image_filter_results, method):
    filter_handle = globals().get("".join([method, "_filter"]))
    expected_results = image_filter_results.get(method)
    # test out-of-place
    results = filter_handle(sample_images)
    np.testing.assert_equal(results, expected_results)
    # test in-place
    # filter_handle(sample_images, in_place=True)
    # np.testing.assert_equal(sample_images, expected_results)
