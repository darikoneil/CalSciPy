import pytest

import numpy as np

from CalSciPy.traces import calculate_dfof, calculate_standardized_noise, detrend_polynomial


"""
Test suite for trace processing functions (generic)

"""


@pytest.mark.parametrize("method", ["low-pass", "mean", "median", "moving_mean", "percentile", "sliding_mean",
                                    "sliding_median", "sliding_percentile"])
def test_calculate_dfof(sample_traces, dfof_results, method):
    # grab expected results
    results = dfof_results.get(method)
    dfof = calculate_dfof(sample_traces, method=method)
    # check out-of-place
    np.testing.assert_equal(dfof, results)
    # check external reference
    ext_dfof = calculate_dfof(sample_traces, method=method, external_reference=sample_traces + 1)
    np.testing.assert_raises(AssertionError, np.testing.assert_equal, ext_dfof, results)
    # check in-place
    calculate_dfof(sample_traces, method=method, in_place=True)
    np.testing.assert_equal(sample_traces, results)


def test_calculate_standardized_noise(sample_traces, standardized_noise_sample_traces):
    std_noise = calculate_standardized_noise(sample_traces, frame_rate=30.0)
    np.testing.assert_equal(std_noise, standardized_noise_sample_traces)


def test_detrend_polynomial(sample_traces, detrended_sample_traces):
    # test out-of-place
    detrended_traces = detrend_polynomial(sample_traces, in_place=False)
    np.testing.assert_equal(detrended_traces, detrended_sample_traces)
    # test in-place
    detrend_polynomial(sample_traces, in_place=True)
    np.testing.assert_equal(sample_traces, detrended_sample_traces)
