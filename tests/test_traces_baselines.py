import pytest

import numpy as np

from CalSciPy.traces.baseline import baseline_calculation


"""
Test suite for baseline calculation methods (traces)

"""


@pytest.mark.parametrize("method", ["low-pass", "mean", "median", "moving_mean", "percentile", "sliding_mean",
                                    "sliding_median", "sliding_percentile"])
def test_baseline_calculations(sample_traces, baseline_results, method):
    # grab expected results
    results = baseline_results.get(method)
    # retrieve function handle. this is makes the test dependent on baseline_calculation function,
    # but the simplicity is worth it
    baseline_function = baseline_calculation(method)
    # calculate
    baseline = baseline_function(sample_traces)
    # check
    np.testing.assert_equal(baseline, results)


def test_unimplemented_baseline_method():
    with pytest.raises(NotImplementedError):
        baseline_calculation("Total Perspective Vortex")
