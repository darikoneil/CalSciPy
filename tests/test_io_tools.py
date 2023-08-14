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

    def check_data(self, image):
        np.testing.assert_array_equal(self.data, image)

    def test_load_binary(self):
        test_data = load_binary(self.directory.joinpath("binary"))
        self.check_data(test_data)

    def test_save_binary(self):
        save_binary(self.outputs, self.data)

    def test_mutation_binary(self):
        save_binary(self.outputs, self.data)
        test_data = load_binary(self.outputs)
        self.check_data(test_data)

    def test_load_single_stack(self):
        ...

    def test_save_single_stack(self):
        ...

    def test_mutation_single_stack(self):
        ...

    def test_load_multi_stack(self):
        ...

    def test_save_multi_stack(self):
        ...

    def test_mutation_multi_stack(self):
        ...


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
        io_helper.test_load_binary()

    def test_save_binary(self, io_helper):
        io_helper.test_save_binary()

    def test_mutation_binary(self, io_helper):
        io_helper.test_mutation_binary()
