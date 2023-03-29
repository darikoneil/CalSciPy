import os
import pytest
from shutil import rmtree
from CalSciPy.io_tools import load_all_tiffs, load_single_tiff
from CalSciPy.bruker import determine_bruker_folder_contents, repackage_bruker_tiffs, load_bruker_tiffs
import numpy as np
import pathlib
from tests.helpers import BlockPrinting, read_descriptions
from PPVD.style import TerminalStyle
from tests.conftest import sample_data_dir


DATASET = pytest.mark.datafiles(
    sample_data_dir,
    keep_top_dir=False,
    on_duplicate="ignore",
)


@DATASET
def test_determine_bruker_folder_contents(datafiles):
    with BlockPrinting():
        for _dir in datafiles.listdir():

            _input_folder = next(pathlib.Path(_dir).glob("bruker_folder"))
            _descriptions = next(pathlib.Path(_dir).glob("description.txt"))
            _descriptions = read_descriptions(_descriptions)

            _contents = determine_bruker_folder_contents(_input_folder)

            for _test in enumerate(["Channel", "Plane", "Frame", "Height", "Width"]):
                assert _contents[_test[0]] == _descriptions[_test[0]], f"Description Mismatch: failed on dataset " \
                                                                       f"{pathlib.Path(_dir).name}: " \
                                                                       f"{_test[1]} detection"
            return

        rmtree(datafiles)


@DATASET
def test_repackage_bruker_tiffs(datafiles, tmp_path):
    with BlockPrinting():
        for _dir in datafiles.listdir():
            # INGEST
            _input_folder = next(pathlib.Path(_dir).glob("bruker_folder"))
            _descriptions = next(pathlib.Path(_dir).glob("description.txt"))
            _output_folder = os.path.join(str(tmp_path), str(pathlib.Path(_dir).stem) + "_output")
            # MAKE OUTPUT FOLDER
            os.mkdir(_output_folder)
            # RUN FUNCTION
            repackage_bruker_tiffs(_input_folder, _output_folder, (0, 0))
            # TEST
            _descriptions = read_descriptions(_descriptions)
            _contents = load_all_tiffs(_output_folder)
            assert _contents.shape[0] == np.cumprod(_descriptions[2])[-1], f"Image Mismatch: failed on dataset" \
                                                                           f"{pathlib.Path(_input_folder).name}"


@DATASET
def test_load_bruker_tiffs(datafiles):
    with BlockPrinting():
        for _dir in datafiles.listdir():
            # INGEST
            _input_folder = next(pathlib.Path(_dir).glob("bruker_folder"))
            _input_image = next(pathlib.Path(_dir).glob("Video_01_of_1.tif"))
            _descriptions = next(pathlib.Path(_dir).glob("description.txt"))
            # LOAD COMPARISON
            _descriptions = read_descriptions(_descriptions)
            _image1 = load_single_tiff(_input_image)
            # TEST
            _image2 = load_bruker_tiffs(_input_folder, 1, 0)[0]
            np.testing.assert_array_equal(_image1, _image2, err_msg=f"Image Mismatch: failed on dataset "
                                                                    f"{pathlib.Path(_input_folder).name}")
