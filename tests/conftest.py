import pytest
from os import environ
from pathlib import Path
import numpy as np


# Adjust base dir if necessary
BASE_DIRECTORY = Path.cwd()
if "tests" not in str(BASE_DIRECTORY):
    BASE_DIRECTORY = BASE_DIRECTORY.joinpath("tests")

SAMPLES_DIRECTORY_ = BASE_DIRECTORY.joinpath("testing_data")
SAMPLES_DATASETS_DIRECTORY = SAMPLES_DIRECTORY_.joinpath("sample_datasets")
SAMPLES_VARIABLES_DIRECTORY = SAMPLES_DIRECTORY_.joinpath("sample_variables")

# Turn off JIT for proper coverage
environ["NUMBA_DISABLE_JIT"] = "1"


@pytest.fixture(scope="function")
def spike_times(request):
    return [[1, 2, 3], [5, 6, 7], [0, 10, 20]], [[1, 2, 3], [5, 6, 7], [0, 10, 20]]


@pytest.fixture(scope="function")
def matrix(request):
    sample_matrix = np.full((5, 100), 1)
    sample_matrix[0, :] = np.arange(100)
    return sample_matrix


@pytest.fixture(scope="function")
def tensor(request):
    return np.array([np.full((4, 5), 1), np.full((4, 5), 2), np.full((4, 5), 3)])


@pytest.fixture(scope="function")
def factorized_matrix(request):
    return np.load(SAMPLES_VARIABLES_DIRECTORY.joinpath("sample_factorized_matrices.npy"), allow_pickle=True)


@pytest.fixture(scope="function")
def sample_traces(request):
    return np.load(SAMPLES_VARIABLES_DIRECTORY.joinpath("sample_traces.npy"), allow_pickle=True)


@pytest.fixture(scope="function")
def extended_sample_traces(request):
    return np.load(SAMPLES_VARIABLES_DIRECTORY.joinpath("extended_sample_traces.npy"), allow_pickle=True)


@pytest.fixture(scope="function")
def dfof_sample_traces(request):
    return np.load(SAMPLES_VARIABLES_DIRECTORY.joinpath("dfof_sample_traces.npy"), allow_pickle=True)


@pytest.fixture(scope="function")
def dfof_offset_sample_traces(request):
    return np.load(SAMPLES_VARIABLES_DIRECTORY.joinpath("dfof_offset_sample_traces.npy"), allow_pickle=True)


@pytest.fixture(scope="function")
def dfof_ext_sample_traces(request):
    return np.load(SAMPLES_VARIABLES_DIRECTORY.joinpath("dfof_ext_sample_traces.npy"), allow_pickle=True)


@pytest.fixture(scope="function")
def dfof_frame_rate_sample_traces(request):
    return np.load(SAMPLES_VARIABLES_DIRECTORY.joinpath("dfof_frame_rate_sample_traces.npy"), allow_pickle=True)


@pytest.fixture(scope="function")
def dfof_small_sample_sample_traces(request):
    return np.load(SAMPLES_VARIABLES_DIRECTORY.joinpath("dfof_small_sample_sample_traces.npy"), allow_pickle=True)


@pytest.fixture(scope="function")
def dfof_even_sample_traces(request):
    return np.load(SAMPLES_VARIABLES_DIRECTORY.joinpath("dfof_even_sample_traces.npy"), allow_pickle=True)


@pytest.fixture(scope="function")
def std_noise_sample_traces(request):
    return np.load(SAMPLES_VARIABLES_DIRECTORY.joinpath("std_noise_sample_traces.npy"), allow_pickle=True)


@pytest.fixture(scope="function")
def std_noise_frame_rate_halved_sample_traces(request):
    return np.load(SAMPLES_VARIABLES_DIRECTORY.joinpath("std_noise_frame_rate_halved_sample_traces.npy"),
                   allow_pickle=True)


@pytest.fixture(scope="function")
def detrended_dfof_sample_traces(request):
    return np.load(SAMPLES_VARIABLES_DIRECTORY.joinpath("detrended_dfof_sample_traces.npy"), allow_pickle=True)


@pytest.fixture(scope="function")
def spike_probabilities(request):
    return np.array([
        [0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 2.20],
        [2.20, 0.75, 0.75, 0.5, 0.5, 0.25, 0.25],
        [0, 0, 0.1, 0.1, 0.75, 1, 1]
    ])


@pytest.fixture(scope="function")
def expected_firing_rate(request):
    return np.array([
        [7.5, 7.5, 15.0, 15.0, 22.5, 22.5, 66.0],
        [66.0, 22.5, 22.5, 15.0, 15.0, 7.5, 7.5],
        [0.0, 0.0, 3.0, 3.0, 22.5, 30.0, 30.0]
    ])


@pytest.fixture(scope="function")
def frame_rate(request):
    return 30.0


@pytest.fixture(scope="function")
def mean_firing_rates(request):
    return np.array([0.74285714, 0.74285714, 0.42142857])


@pytest.fixture(scope="function")
def normalized_firing_rates(request):
    return np.array([
        [0.11364, 0.33333, 0.66667, 1.00000, 1.00000, 0.75000, 1.00000],
        [1.00000, 1.00000, 1.00000, 1.00000, 0.66667, 0.25000, 0.11364],
        [0.00000, 0.00000, 0.13333, 0.20000, 1.00000, 1.00000, 0.45455]
    ])
