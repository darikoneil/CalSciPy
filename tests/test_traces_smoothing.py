import pytest

import numpy as np

from CalSciPy.traces import perona_malik_diffusion


"""
Test suite for smoothing functions used on traces
"""


def test_perona(sample_traces, perona_smoothed_sample_traces):
    # test out-of-place
    smoothed = perona_malik_diffusion(sample_traces, in_place=False)
    np.testing.assert_allclose(smoothed, perona_smoothed_sample_traces, atol=1e-5)
    # test in-place
    perona_malik_diffusion(sample_traces, in_place=True)
    np.testing.assert_allclose(sample_traces, perona_smoothed_sample_traces, atol=1e-5)
