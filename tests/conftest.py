import pytest
# noinspection PyProtectedMember
from CalSciPy._helpers import copy_dummies, purge_dummies, identify_dummy_source

from os import environ
from pathlib import Path

import numpy as np

"""
CONFIGURATION FOR TESTING
"""

# Turn off JIT for proper coverage if anything uses numba
environ["NUMBA_DISABLE_JIT"] = "1"


# COPY DUMMIES WITH EACH TEST TO AVOID MUTATION CONCERNS
_DUMMY_SOURCE_DIRECTORY = identify_dummy_source()

# MANUALLY MAKE TEMP DIRECTORY SINCE PYTEST'S TMP_PATH IS SOLELY FUNCTION SCOPE
_TEMPORARY_DIRECTORY = Path().cwd().joinpath("temp")
if not _TEMPORARY_DIRECTORY.exists():
    _TEMPORARY_DIRECTORY.mkdir(exist_ok=True)

# MANUALLY ADD THE THE SET OF SAMPLES THE FIRST TIME
copy_dummies(_DUMMY_SOURCE_DIRECTORY, _TEMPORARY_DIRECTORY)


@pytest.fixture(scope="session")
def temp_path():
    return _TEMPORARY_DIRECTORY


# Call only for tests that risk manipulating the original datafiles
@pytest.fixture(autouse=False)
def datafiles():

    # USING TEMP_PATH INSTEAD OF PYTEST TMP_PATH BECAUSE THAT IS A FUNCTION-BASED FIXTURE

    # COPY DUMMIES
    copy_dummies(_DUMMY_SOURCE_DIRECTORY, _TEMPORARY_DIRECTORY)

    # TEST STUFF
    yield

    # PURGE
    purge_dummies(_TEMPORARY_DIRECTORY)


# GENERATORS FOR ITERATING THROUGH INDIVIDUAL SAMPLE SETS FOR TESTING
# EACH SAMPLE SET GROUP (SUB-FOLDER) HAS A FUNCTION TO ACT AS A GENERATOR
def retrieve_dataset_ids() -> Path:
    """
    Generator that yields the next dataset in the data directory sample set
    """

    # get the sub-folder containing data directory samples
    data_dir = _TEMPORARY_DIRECTORY.joinpath("datasets")

    # generator for testing
    return [data.name for data in data_dir.glob("*")]


def retrieve_roi() -> dict:
    """
    Generator that yields the next sample roi in the data directory sample set
    """
    sample_rois_file = _TEMPORARY_DIRECTORY.joinpath("variables").joinpath("sample_rois.npy")

    sample_rois = np.load(sample_rois_file, allow_pickle=True).item()

    for roi in sample_rois.values():
        yield roi


def retrieve_suite2p() -> Path:
    """
    Generator that yields the next dataset in the suite2p sample set
    """
    # get the sub-folder containing data directory samples
    suite2p_dir = _TEMPORARY_DIRECTORY.joinpath("suite2p")

    # generator for testing
    return [suite2p_folder.name for suite2p_folder in suite2p_dir.glob("*")]


# HERE ARE SOME MISCELLANEOUS DATA STRUCTURES REQUIRED BY SEVERAL TEST SUITES
@pytest.fixture(scope="function")
def sample_factorized_matrix(request):
    return np.load(_TEMPORARY_DIRECTORY.joinpath("variables").joinpath("sample_factorized_matrices.npy"),
                   allow_pickle=True)


@pytest.fixture(scope="function")
def sample_matrix(request):
    sample_matrix = np.full((5, 100), 1)
    for row in range(0, 5, 2):
        sample_matrix[row, :] = np.arange(100)
    return sample_matrix


@pytest.fixture(scope="function")
def sample_reference_image(request):
    return np.load(_TEMPORARY_DIRECTORY.joinpath("variables").joinpath("sample_reference_image.npy"), allow_pickle=True)


@pytest.fixture(scope="function")
def sample_roi_map(request):
    return np.load(_TEMPORARY_DIRECTORY.joinpath("variables").joinpath("sample_roi_map.npy"), allow_pickle=True).item()


@pytest.fixture(scope="function")
def sample_tensor(request):
    sample_matrix = np.full((5, 100), 1)
    for row in range(0, 5, 2):
        sample_matrix[row, :] = np.arange(100)
    sample_tensor = np.zeros((10, 5, 10))
    for i in range(10):
        sample_tensor[i, :, :] = sample_matrix[:, i * 10: (i + 1) * 10]
    return sample_tensor


@pytest.fixture(scope="function")
def sample_traces(request):
    return np.load(_TEMPORARY_DIRECTORY.joinpath("variables").joinpath("sample_traces.npy"))


# HERE ARE SOME HARDCODED RESULTS


@pytest.fixture(scope="session")
def detrended_sample_traces(request):
    return np.load(_TEMPORARY_DIRECTORY.joinpath("results").joinpath("detrended_polynomial_results.npy"),
                   allow_pickle=True)


@pytest.fixture(scope="session")
def dfof_results(request):
    return np.load(_TEMPORARY_DIRECTORY.joinpath("results").joinpath("dfof_results.npy"),
                   allow_pickle=True).item()


@pytest.fixture(scope="session")
def perona_smoothed_sample_traces(request):
    return np.load(_TEMPORARY_DIRECTORY.joinpath("results").joinpath("perona_results.npy"),
                   allow_pickle=True)


@pytest.fixture(scope="session")
def standardized_noise_sample_traces(request):
    return np.load(_TEMPORARY_DIRECTORY.joinpath("results").joinpath("std_noise_results.npy"),
                   allow_pickle=True)

