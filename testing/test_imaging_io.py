import os
import pytest
from shutil import rmtree
from imaging.io import determine_bruker_folder_contents, load_all_tiffs, load_bruker_tiffs, load_single_tiff, \
    repackage_bruker_tiffs
import numpy as np
import pathlib


FIXTURE_DIR = "".join([os.path.abspath(os.path.join(os.getcwd(), os.pardir)), "\\testing_data"])

DATASET = pytest.mark.datafiles(
    "".join([FIXTURE_DIR, "\\sample_datasets"]),
    keep_top_dir=False,
    on_duplicate="ignore",
)


def read_descriptions(file):
    return np.genfromtxt(file, delimiter=",", dtype="int")


@DATASET
def test_determine_bruker_folder_contents(datafiles):
    for _dir in datafiles.listdir():
        _dataset, _descriptions = [_folder for _folder in pathlib.Path(_dir).glob("*")
                                   if _folder.stem == "bruker_folder" or "descriptions"]
        _descriptions = read_descriptions(_descriptions)

        _contents = determine_bruker_folder_contents(_dataset)

        for _test in enumerate(["Channel", "Plane", "Frame", "Height", "Width"]):
            assert _contents[_test[0]] == _descriptions[_test[0]], f"Failed On {pathlib.Path(_dir).name}: " \
                                                           f"{_test[1]} Detection"
        return
    rmtree(datafiles)


@DATASET
def test_repackage_bruker_tiffs(datafiles, tmp_path):
    for _dir in datafiles.listdir():
        # INGEST
        _input_folder, _descriptions = [_folder for _folder in pathlib.Path(_dir).glob("*")
                         if _folder.stem == "bruker_folder" or "descriptions"]
        _output_folder_full = "".join([str(tmp_path), str(_input_folder.stem), "_output_full"])
        _output_folder_single = "".join([str(tmp_path), str(_input_folder.stem), "_output_single"])
        # MAKE OUTPUT FOLDER
        os.mkdir(_output_folder_full)
        os.mkdir(_output_folder_single)
        # RUN FUNCTION
        repackage_bruker_tiffs(_input_folder, _output_folder_full)
        repackage_bruker_tiffs(_input_folder, _output_folder_single, (0, 0)) # Zero plane, Zero Channel
        # TEST FULL
        _descriptions = read_descriptions(_descriptions)
        _contents = load_all_tiffs(_output_folder_full)
        assert _images.shape[0] == np.cumprod(_descriptions[0:3]), f"Failed On {pathlib.Path(_input_folder).name} " \
                                                                   f"Full-Repackage"
        # TEST SINGLE
        _contents = load_all_tiffs(_output_folder_single)
        assert _images.shape[0] == _descriptions[2], f"Failed On {pathlib.Path(_input_folder).name} Single-Repackage"

    rmtree(datafiles)


@DATASET
def test_load_bruker_tiffs(datafiles):
    for _dir in datafiles.listdir():
        # INGEST
        _dataset, _descriptions = [_folder for _folder in pathlib.Path(_dir).glob("*")
                                   if _folder.stem == "bruker_folder" or "descriptions"]
        # GET COMPARISON DESCRIPTIONS
        _descriptions = read_descriptions(_descriptions)
        _contents = determine_bruker_folder_contents(_dataset)
        # FULL TEST
        _images = load_bruker_tiffs(_dataset)
        print(f"{_images.shape}")
        print(f"{np.cumprod(_descriptions[0:3])}")
        # assert _images.shape[0] == np.cumprod(_descriptions[0:3]), f"Failed On {pathlib.Path(_input_folder).name} " \
         #                                                           f"Full-Repackage"
    rmtree(datafiles)


@DATASET
def test_load_single_tiff(datafiles):
    # INGEST
    for _dir in datafiles.listdir():
        _dataset, _descriptions = [_folder for _folder in pathlib.Path(_dir).glob("*")
                                   if _folder.stem == "bruker_folder" or "descriptions"]
        _file = [_file for _file in pathlib.Path(_dataset).glob("*.tif")][0]
        # GET COMPARISON DESCRIPTIONS
        _descriptions = read_descriptions(_descriptions)
        _contents = determine_bruker_folder_contents(_dataset)
        # TEST
        _image = load_single_tiff(_file, 1)
        np.testing.assert_array_equal(_image.shape, _descriptions[3:5], err_msg=
        f"Failed On {pathlib.Path(_dir).name}")
    rmtree(datafiles)
