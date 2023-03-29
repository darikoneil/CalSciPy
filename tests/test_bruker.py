import pytest
import numpy as np
from pathlib import Path
from shutil import rmtree
from tests.helpers import BlockPrinting, read_descriptions
from tests.conftest import SAMPLES_DATASETS_DIRECTORY
from CalSciPy.io_tools import load_all_tiffs, load_single_tiff
from CalSciPy.bruker import determine_imaging_content, load_bruker_tiffs, repackage_bruker_tiffs


DATASET = pytest.mark.datafiles(
    SAMPLES_DATASETS_DIRECTORY,
    keep_top_dir=False,
    on_duplicate="ignore",
)


@DATASET
def test_determine_imaging_content(datafiles):
    with BlockPrinting():
        for _dir in datafiles.listdir():

            _input_folder = next(Path(_dir).glob("bruker_folder"))
            _descriptions = next(Path(_dir).glob("description.txt"))
            _descriptions = read_descriptions(_descriptions)

            _contents = determine_imaging_content(_input_folder)

            for _test in enumerate(["Channel", "Plane", "Frame", "Height", "Width"]):
                assert _contents[_test[0]] == _descriptions[_test[0]], f"Description Mismatch: failed on dataset " \
                                                                       f"{Path(_dir).name}: " \
                                                                       f"{_test[1]} detection"
            return

        rmtree(datafiles)


@DATASET
def test_repackage_bruker_tiffs(datafiles, tmp_path):
    with BlockPrinting():
        for _dir in datafiles.listdir():
            # INGEST
            _input_folder = next(Path(_dir).glob("bruker_folder"))
            _descriptions = next(Path(_dir).glob("description.txt"))
            _output_folder = Path(tmp_path).joinpath("".join([Path(_dir).stem, "_output"]))
            # MAKE OUTPUT FOLDER
            _output_folder.mkdir(parents=True, exist_ok=True)
            # RUN FUNCTION
            repackage_bruker_tiffs(_input_folder, _output_folder, (0, 0))
            # TEST
            _descriptions = read_descriptions(_descriptions)
            _contents = load_all_tiffs(_output_folder)
            assert _contents.shape[0] == np.cumprod(_descriptions[2])[-1], f"Image Mismatch: failed on dataset" \
                                                                           f"{_input_folder.name}"


@DATASET
def test_load_bruker_tiffs(datafiles):
    with BlockPrinting():
        for _dir in datafiles.listdir():
            # INGEST
            _input_folder = next(Path(_dir).glob("bruker_folder"))
            _input_image = next(Path(_dir).glob("Video_01_of_1.tif"))
            _descriptions = next(Path(_dir).glob("description.txt"))
            # LOAD COMPARISON
            _descriptions = read_descriptions(_descriptions)
            _image1 = load_single_tiff(_input_image)
            # TEST
            _image2 = load_bruker_tiffs(_input_folder, 1, 0)[0]
            np.testing.assert_array_equal(_image1, _image2, err_msg=f"Image Mismatch: failed on dataset "
                                                                    f"{_input_folder.name}")
