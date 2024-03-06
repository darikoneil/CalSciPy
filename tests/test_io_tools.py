from __future__ import annotations
from typing import Callable, Any

import pytest
from .conftest import retrieve_dataset_ids
# noinspection PyProtectedMember
from CalSciPy._helpers import read_descriptions, generate_dummy_file_name, generate_dummy_output_folder, BlockPrinting

from pathlib import Path
import numpy as np

# noinspection PyProtectedMember
from CalSciPy.io_tools import load_images, _load_single_tif, _load_many_tif, save_images, _save_single_tif, \
    _save_many_tif, load_binary, save_binary, load_video, save_video, load_gif, save_gif


"""
This test suite is designed to test the io-tools module
"""


"""
JUSTIFICATION FOR CODE WITHOUT DIRECT TESTING:
    * _Metadata is sufficiently tested through direct calls from load/save binary
    * check_filepath is sufficiently tested through calls from every save function
"""


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

    def check_data(self, image: np.ndarray, subset: Tuple[int, int] = None, dtype: Any = None):
        if dtype:
            np.testing.assert_allclose(self.data.astype(dtype), image, atol=1.1)
        elif subset is not None:
            np.testing.assert_array_equal(self.data[subset[0]:subset[1], ...], image)
        else:
            np.testing.assert_array_equal(self.data, image)

    def load_validator(self, function: Callable):

        # Try folder that exists but has no imaging files
        with pytest.raises(FileNotFoundError):
            function(Path.cwd())

        # must also fail general validation tests
        with pytest.raises(FileNotFoundError):
            function(Path.cwd().joinpath("arthur_dent"))
        with pytest.raises(ValueError):
            function("C:\\ford_prefect ***%")  # FAIL PERMITTED CHARS
        with pytest.raises(TypeError):
            # noinspection PyTypeChecker
            function(125.6)  # fail with bad type so we don't do anything unexpected

    def save_validator(self, function: Callable):
        with pytest.raises(ValueError):
            function("C:\\vogon_destructor_fleet &^6*", self.data)  # FAIL VALIDATE PATH PERMITTED CHARS
        with pytest.raises(TypeError):
            # noinspection PyTypeChecker
            function(125.6, self.data)  # fail with bad type so we don't do anything unexpected


@pytest.fixture()
def io_helper(request, temp_path):
    """
    Fixture for the helper class

    """
    return IO(request.param, temp_path)


#@pytest.mark.usefixtures("datafiles")
@pytest.mark.parametrize("io_helper", [dataset for dataset in retrieve_dataset_ids()], indirect=["io_helper"])
class TestIO:
    """
    Actual test class

    """
    def test_load_binary(self, io_helper):
        # standard load
        test_data = load_binary(io_helper.directory.joinpath("binary"))
        io_helper.check_data(test_data)
        # standard load but with exact file name
        test_data = load_binary(io_helper.directory.joinpath("binary").joinpath("binary_video"))
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
        # overwrite folder
        save_binary(io_helper.outputs, io_helper.data)
        # exact save
        save_binary(io_helper.outputs, io_helper.data, "exact_filename")
        # overwrite exact save
        save_binary(io_helper.outputs.joinpath("exact_filename").with_suffix(".bin"), io_helper.data)
        # save to folder that doesn't exist yet
        save_binary(io_helper.outputs.joinpath("slartibartfast"), io_helper.data)

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

    def test_load_single_image(self, io_helper):
        # implementation
        test_data = _load_single_tif(io_helper.directory.joinpath("single_page").joinpath("images.tif"))
        io_helper.check_data(test_data, subset=(0, 1))
        # standard
        test_data = load_images(io_helper.directory.joinpath("single_page"))
        io_helper.check_data(test_data, subset=(0, 1))
        # exact
        test_data = load_images(io_helper.directory.joinpath("single_page").joinpath("images.tif"))
        io_helper.check_data(test_data, subset=(0, 1))

    def test_save_single_image(self, io_helper):
        # implementation (1, ...) shape for frames x height x width
        _save_single_tif(io_helper.outputs.joinpath("single_page"), io_helper.data[0:1, ...])
        # implementation (...) shape for height x width
        _save_single_tif(io_helper.outputs.joinpath("single_page"), np.reshape(io_helper.data[0:1, ...],
                                                                               io_helper.data.shape[1:])
                         )
        # standard
        save_images(io_helper.outputs.joinpath("single_page"), io_helper.data[0:1, ...])
        # exact
        save_images(io_helper.outputs.joinpath("single_page"), io_helper.data[0:1, ...], "exact_filename")

    def test_mutation_single_image(self, io_helper):
        save_images(io_helper.outputs.joinpath("single_page"), io_helper.data[0:1, ...])
        test_data = load_images(io_helper.outputs.joinpath("single_page"))
        io_helper.check_data(test_data, subset=(0, 1))

    def test_single_image_exceptions(self, io_helper):
        # Redundant, included to organizational purposes
        io_helper.load_validator(load_images)
        io_helper.save_validator(save_images)

    def test_load_single_stack(self, io_helper):
        # test implementation function
        test_data = _load_single_tif(io_helper.directory.joinpath("single_stack").joinpath("images.tif"))
        io_helper.check_data(test_data)
        # test standard load
        test_data = load_images(io_helper.directory.joinpath("single_stack"))
        io_helper.check_data(test_data)
        # exact
        test_data = load_images(io_helper.directory.joinpath("single_stack").joinpath("images.tif"))
        io_helper.check_data(test_data)

    def test_save_single_stack(self, io_helper):
        # test implementation function
        _save_single_tif(io_helper.outputs.joinpath("single_stack"), io_helper.data)
        # test standard save
        save_images(io_helper.outputs.joinpath("single_stack"), io_helper.data)
        # test exact save
        save_images(io_helper.outputs, io_helper.data, "exact_filename")

    def test_mutation_single_stack(self, io_helper):
        # standard load-save with check for mutation
        save_images(io_helper.outputs.joinpath("single_stack.tif"), io_helper.data)
        test_data = load_images(io_helper.outputs.joinpath("single_stack.tif"))
        io_helper.check_data(test_data)

    def test_single_stack_exceptions(self, io_helper):
        io_helper.load_validator(load_images)
        io_helper.save_validator(save_images)

    def test_load_multi_stack(self, io_helper):
        # test implementation function
        test_data = _load_many_tif(io_helper.directory.joinpath("multi_stack"))
        io_helper.check_data(test_data)
        # test standard load
        test_data = load_images(io_helper.directory.joinpath("multi_stack"))
        io_helper.check_data(test_data)

    def test_save_multi_stack(self, io_helper):
        # test implementation function
        _save_many_tif(io_helper.outputs.joinpath("multi_stack"), io_helper.data, size_cap=0.005)
        # test standard function
        save_images(io_helper.outputs.joinpath("multi_stack"), io_helper.data, size_cap=0.005)
        # test exact
        save_images(io_helper.outputs.joinpath("multi_stack"), io_helper.data, "exact_filename", size_cap=0.005)

    def test_mutation_multi_stack(self, io_helper):
        save_images(io_helper.outputs.joinpath("multi_stack"), io_helper.data, size_cap=0.005)
        test_data = load_images(io_helper.outputs.joinpath("multi_stack"))
        io_helper.check_data(test_data)

    def test_multi_stack_exceptions(self, io_helper):
        # Redundant, included to organizational purposes
        io_helper.load_validator(load_images)
        io_helper.save_validator(save_images)

    def test_load_video(self, io_helper):
        test_data = load_video(io_helper.directory.joinpath("video").joinpath("video.mp4"))
        io_helper.check_data(test_data[:, :, :, 0], dtype=test_data.dtype)

    def test_save_video(self, io_helper):
        with BlockPrinting():
            save_video(io_helper.outputs.joinpath("video"), io_helper.data)

    def test_mutation_video(self, io_helper):
        with BlockPrinting():
            save_video(io_helper.outputs.joinpath("video"), io_helper.data)
        test_data = load_video(io_helper.outputs.joinpath("video").joinpath("video"))
        io_helper.check_data(test_data[:, :, :, 0], dtype=test_data.dtype)

    def test_video_exceptions(self, io_helper):
        # Redundant, included to organizational purposes
        io_helper.load_validator(load_video)
        io_helper.save_validator(save_video)

    @pytest.mark.skip(reason="Expected Failure")
    def test_load_gif(self, io_helper):
        test_data = load_gif(io_helper.directory.joinpath("gif").joinpath("images.gif"))
        io_helper.check_data(test_data[:, :, :, 0], dtype=test_data.dtype)

    def test_save_gif(self, io_helper):
        with BlockPrinting():
            save_gif(io_helper.outputs.joinpath("gif"), io_helper.data)

    @pytest.mark.skip(reason="Expected Failure")
    def test_mutation_gif(self, io_helper):
        with BlockPrinting():
            save_gif(io_helper.outputs.joinpath("gif"), io_helper.data)
        test_data = load_gif(io_helper.outputs.joinpath("gif").joinpath("images.gif"))
        io_helper.check_data(test_data[:, :, :, 0], dtype=test_data.dtype)

    def test_gif_exceptions(self, io_helper):
        # Redundant, included to organizational purposes
        io_helper.save_validator(save_gif)
