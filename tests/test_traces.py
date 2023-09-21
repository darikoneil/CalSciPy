import pytest

import numpy as np

from CalSciPy.traces import calculate_standardized_noise, detrend_polynomial


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
