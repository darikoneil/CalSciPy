import pytest
from tests.helpers import copy_dummies, purge_dummies, identify_dummy_source

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


@pytest.fixture(autouse=True)
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


# HERE ARE SOME MISCELLANEOUS DATA STRUCTURES REQUIRED BY SEVERAL TEST SUITES
@pytest.fixture(scope="function")
def matrix(request):
    sample_matrix = np.full((5, 100), 1)
    for row in range(0, 5, 2):
        sample_matrix[row, :] = np.arange(100)
    return sample_matrix
