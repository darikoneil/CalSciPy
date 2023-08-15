from __future__ import annotations
from typing import Callable

import pytest
from tests.conftest import retrieve_dataset_ids
from tests.helpers import read_descriptions, generate_dummy_file_name, generate_dummy_output_folder

from pathlib import Path
import numpy as np

# noinspection PyProtectedMember
from CalSciPy.io_tools import load_images, _load_single_tif, _load_many_tif, save_images, _save_single_tif, \
    _save_many_tif, load_binary, save_binary


class IO:
    """
    Helper class for io tests for an example dataset
    """

    def __init__(self, dataset: str, temp_path: Path):
        # directory of the test dataset
        self.directory = generate_dummy_file_name(dataset, temp_path, "datasets")
        # temp path
        self.outputs = generate_dummy_output_folder(dataset, temp_path)
        # characteristics of the test data
        self.descriptions = read_descriptions(self.directory.joinpath("description.txt"))
        # actual data
        self.data = \
            np.fromfile(self.directory.joinpath("binary").joinpath("binary_video").with_suffix(".bin"), dtype=np.uint16)
        self.data = np.reshape(self.data, self.descriptions[2:])

    def check_data(self, image: np.ndarray):
        np.testing.assert_array_equal(self.data, image)

    def load_validator(self, function: Callable):

        # must fail general validation tests
        with pytest.raises(FileNotFoundError):
            function(Path.cwd().joinpath("arthur_dent"))
        with pytest.raises(ValueError):
            function("C:\\ford_prefect ***%")  # FAIL PERMITTED CHARS
        with pytest.raises(TypeError):
            # noinspection PyTypeChecker
            function(125.6)  # fail with bad type so we don't do anything unexpected

    def save_validator(self, function: Callable):
        with pytest.raises(ValueError):
            function("C:\\vogon_destructor_fleet &^6*", matrix)  # FAIL VALIDATE PATH PERMITTED CHARS
        with pytest.raises(TypeError):
            # noinspection PyTypeChecker
            function(125.6, matrix)  # fail with bad type so we don't do anything unexpected


@pytest.fixture()
def io_helper(request, temp_path):
    """
    Fixture for the helper class

    """
    return IO(request.param, temp_path)


@pytest.mark.parametrize("io_helper", [dataset for dataset in retrieve_dataset_ids()], indirect=["io_helper"])
class TestIO:
    """
    Actual test class

    """
    def test_load_binary(self, io_helper):
        # standard load
        test_data = load_binary(io_helper.directory.joinpath("binary"))
        io_helper.check_data(test_data)
        # memory map load
        test_data = load_binary(io_helper.directory.joinpath("binary"), mapped=True)
        io_helper.check_data(test_data)
        # interpolate metadata load
        test_data = load_binary(io_helper.directory.joinpath("binary"),
                                missing_metadata={"dtype": np.uint16,
                                                  "y": io_helper.descriptions[3],
                                                  "x": io_helper.descriptions[4]},
                                )
        io_helper.check_data(test_data)

    def test_save_binary(self, io_helper):
        # normal save
        save_binary(io_helper.outputs, io_helper.data)
        # exact save
        save_binary(io_helper.outputs, io_helper.data, "exact_filename")

    def test_mutation_binary(self, io_helper):
        # standard load-save with check for mutation
        save_binary(io_helper.outputs, io_helper.data)
        test_data = load_binary(io_helper.outputs)
        io_helper.check_data(test_data)

    def test_binary_exceptions(self, io_helper):
        # fail if no dtype provided
        with pytest.raises(AttributeError):
            load_binary(io_helper.directory.joinpath("binary"),
                        missing_metadata={"frames": io_helper.descriptions[2],
                                          "y": io_helper.descriptions[3],
                                          "x": io_helper.descriptions[4]
                                          }
                        )
        # fail if only 1 of 3 shape provided
        with pytest.raises(AttributeError):
            load_binary(io_helper.directory.joinpath("binary"),
                        missing_metadata={"dtype": np.uint16,
                                          "frames": io_helper.descriptions[2]
                                          }
                        )

        # check for general exceptions that are expected
        io_helper.load_validator(load_binary)
        io_helper.save_validator(save_binary)

    def test_load_single_stack(self, io_helper):
        ...
