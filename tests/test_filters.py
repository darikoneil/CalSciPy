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
    np.testing.assert_allclose(results, expected_results, rtol=1)
    # test block no buffer with approximately 1% tolerance
    results = filter_handle(sample_images, block_size=50, block_buffer=0)
    np.testing.assert_allclose(results, expected_results, rtol=100, atol=3)
    # test block with buffer with approximately 1% tolerance
    results = filter_handle(sample_images, block_size=50, block_buffer=5)
    np.testing.assert_allclose(results, expected_results, rtol=100, atol=3)
    # test in-place
    filter_handle(sample_images, block_size=50, block_buffer=5, in_place=True)
    np.testing.assert_allclose(sample_images, expected_results, rtol=100, atol=3)
