import pytest
from shutil import rmtree
from pathlib import Path
import numpy as np

from tests.conftest import SAMPLES_DATASETS_DIRECTORY
from tests.helpers import read_descriptions
from CalSciPy.io_tools import load_single_tiff, save_single_tiff

DATASET = pytest.mark.datafiles(
    SAMPLES_DATASETS_DIRECTORY,
    keep_top_dir=False,
    on_duplicate="ignore",
)


@DATASET
def test_single_tiff(datafiles, tmp_path):
    for _dir in datafiles.listdir():
        _input_image = next(Path(_dir).glob("single.tif"))
        _descriptions = next(Path(_dir).glob("description.txt"))
        _output_folder = Path(tmp_path).joinpath("".join([Path(_dir).stem, "_output"]))
        # MAKE OUTPUT FOLDER
        Path.mkdir(_output_folder, parents=True, exist_ok=True)
        _output_file = _output_folder.joinpath("single.tif")
        # GET COMPARISON DESCRIPTIONS
        _descriptions = read_descriptions(_descriptions)
        # TEST
        _image = load_single_tiff(_input_image)
        np.testing.assert_array_equal(_image.shape, [1, *_descriptions[3:5]], err_msg=f"Mismatch: failed on dataset"
                                                                                      f" {Path(_dir).name}"
                                                                                      f" during first loading")
        save_single_tiff(_image, _output_file)
        _image2 = load_single_tiff(_output_file)
        np.testing.assert_array_equal(_image, _image2, err_msg=f"Image Mismatch: failed on dataset"
                                                               f" {Path(_dir).name} during second loading")

    # test exceptions
    with pytest.raises(ValueError):
        load_single_tiff("C:\\&^6* ***%")  # FAIL PERMITTED CHARS
    with pytest.raises(FileNotFoundError):
        load_single_tiff("C:\\file_not_exists")  # fail file not exists
    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        load_single_tiff(125.6)  # fail with bad type so we don't do anything unexpected
    with pytest.raises(ValueError):
        save_single_tiff([1, 2, 3, 4, 5], "C:\\&^6*")  # FAIL VALIDATE PATH PERMITTED CHARS
    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        save_single_tiff([1, 2, 3, 4, 5], 125.6)  # fail with bad type so we don't do anything unexpected
    with pytest.raises(ValueError):
        save_single_tiff([1, 2, 3, 4, 5], "C:\\file.mp4")  # FAIL EXTENSION
    # the rest of failures are fine being trial by forgiveness

    rmtree(tmp_path)
