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


