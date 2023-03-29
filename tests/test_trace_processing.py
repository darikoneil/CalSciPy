import pytest
from pytest import approx
import numpy as np
from CalSciPy.trace_processing import calculate_dfof, calculate_standardized_noise, detrend_polynomial


def test_calculate_dfof(sample_traces, extended_sample_traces, dfof_sample_traces, dfof_offset_sample_traces,
                        dfof_ext_sample_traces, dfof_frame_rate_sample_traces, dfof_small_sample_sample_traces,
                        dfof_even_sample_traces):
    # standard
    dfof = calculate_dfof(sample_traces)
    np.testing.assert_allclose(dfof, dfof_sample_traces, rtol=1e-3, err_msg="Failed calculating standard dfof")
    # offset of 1.0; simply a property test here
    dfof_offset = calculate_dfof(sample_traces, offset=1.0)
    np.testing.assert_allclose(dfof_offset, dfof_offset_sample_traces, rtol=1e-3,
                               err_msg="Failed calculating offset dfof")
    # external reference
    dfof_ext = calculate_dfof(sample_traces, external_reference=extended_sample_traces)
    np.testing.assert_allclose(dfof_ext, dfof_ext_sample_traces, rtol=1e-3,
                               err_msg="Failed calculating external reference dfof")
    # frame rate
    dfof_frame_rate = calculate_dfof(sample_traces, frame_rate=15.0)
    np.testing.assert_allclose(dfof_frame_rate, dfof_frame_rate_sample_traces, rtol=1e-3,
                               err_msg="Failed calculating halved-frame rate dfof")
    # small sample
    dfof_small_sample = calculate_dfof(sample_traces[:, 0:50])
    np.testing.assert_allclose(dfof_small_sample, dfof_small_sample_sample_traces, rtol=1e-3,
                               err_msg="Failed on small sample")
    # even sample
    dfof_even_sample = calculate_dfof(sample_traces[:, 0:81])
    np.testing.assert_allclose(dfof_even_sample, dfof_even_sample_traces, rtol=1e-3, err_msg="Failed on even sample")
    # in place
    calculate_dfof(sample_traces, in_place=True)
    np.testing.assert_allclose(sample_traces, dfof_sample_traces, rtol=1e-3,
                               err_msg="Failed calculating in-place dfof")


def test_calculate_standardized_noise(dfof_sample_traces, std_noise_sample_traces,
                                      std_noise_frame_rate_halved_sample_traces):

    std_noise = calculate_standardized_noise(dfof_sample_traces)
    np.testing.assert_allclose(std_noise, std_noise_sample_traces, rtol=1e-3,
                               err_msg="Failed first call to calculate std noise")

    std_noise_frame_rate = calculate_standardized_noise(dfof_sample_traces, 15.0)
    np.testing.assert_allclose(std_noise_frame_rate, std_noise_frame_rate_halved_sample_traces, rtol=1e-3,
                               err_msg="Failed second call to calculate std noise")


def test_polynomial_detrending(dfof_sample_traces, detrended_dfof_sample_traces):
    # out of place
    detrended_dfof = detrend_polynomial(dfof_sample_traces)
    np.testing.assert_allclose(detrended_dfof, detrended_dfof_sample_traces, err_msg="Failed polynomial detrending")
    # in place
    detrend_polynomial(dfof_sample_traces, in_place=True)
    np.testing.assert_allclose(detrended_dfof, dfof_sample_traces, err_msg="Failed in-place polynomial detrending")
