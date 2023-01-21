import os
import pytest
from shutil import rmtree
from src.CalSciPy.io import determine_bruker_folder_contents, load_all_tiffs, load_single_tiff, \
    repackage_bruker_tiffs, save_raw_binary, load_raw_binary, save_single_tiff, save_tiff_stack, \
    save_video, load_bruker_tiffs
import numpy as np
import pathlib


FIXTURE_DIR = "".join([os.path.abspath(os.path.join(os.getcwd(), os.pardir)), "\\testing_data"])

DATASET = pytest.mark.datafiles(
    "".join([FIXTURE_DIR, "\\sample_datasets"]),
    keep_top_dir=False,
    on_duplicate="ignore",
)


def read_descriptions(file):
    return np.genfromtxt(str(file), delimiter=",", dtype="int")


@DATASET
def test_determine_bruker_folder_contents(datafiles):
    for _dir in datafiles.listdir():

        _input_folder = next(pathlib.Path(_dir).glob("bruker_folder"))
        _descriptions = next(pathlib.Path(_dir).glob("description.txt"))
        _descriptions = read_descriptions(_descriptions)

        _contents = determine_bruker_folder_contents(_input_folder)

        for _test in enumerate(["Channel", "Plane", "Frame", "Height", "Width"]):
            assert _contents[_test[0]] == _descriptions[_test[0]], f"Failed On {pathlib.Path(_dir).name}: " \
                                                           f"{_test[1]} Detection"
        return

    rmtree(datafiles)


@DATASET
def test_single_tiff_load_and_save(datafiles, tmp_path):
    # INGEST
    for _dir in datafiles.listdir():
        _input_image = next(pathlib.Path(_dir).glob("single.tif"))
        _descriptions = next(pathlib.Path(_dir).glob("description.txt"))
        _output_folder = "".join([str(tmp_path), "\\", str(pathlib.Path(_dir).stem), "_output"])
        # MAKE OUTPUT FOLDER
        os.mkdir(_output_folder)
        _output_file = "".join([str(tmp_path), "\\", str(pathlib.Path(_dir).stem), "_output\\single.tif"])
        # GET COMPARISON DESCRIPTIONS
        _descriptions = read_descriptions(_descriptions)
        # TEST
        _image = load_single_tiff(_input_image, 1)
        np.testing.assert_array_equal(_image.shape, _descriptions[3:5], err_msg=f"Failed On "
                                                                                f"{pathlib.Path(_dir).name} "
                                                                                f"on first loading")
        save_single_tiff(_image, _output_file)
        _image2 = load_single_tiff(_output_file, 1)
        np.testing.assert_array_equal(_image, _image2, err_msg=f"Failed On "
                                                                                f"{pathlib.Path(_dir).name} "
                                                                                f"on second loading")

    rmtree(tmp_path)


@DATASET
def test_binaries_load_and_save(datafiles, tmp_path):
    for _dir in datafiles.listdir():
        # INGEST
        _input_image = next(pathlib.Path(_dir).glob("Video_01_of_1.tif"))
        _descriptions = next(pathlib.Path(_dir).glob("description.txt"))
        _output_folder = "".join([str(tmp_path), "\\", str(pathlib.Path(_dir).stem), "_output"])
        # MAKE OUTPUT FOLDER
        os.mkdir(_output_folder)
        # GET COMPARISON
        _descriptions = read_descriptions(_descriptions)
        # TEST
        _images1 = load_single_tiff(_input_image, _descriptions[2])
        np.testing.assert_array_equal(_images1.shape, _descriptions[2:5], err_msg=f"Failed On "
                                                                                f"{pathlib.Path(_dir).name} "
                                                                                f"on first loading")
        save_raw_binary(_images1, _output_folder)
        _images2 = load_raw_binary(_output_folder)
        np.testing.assert_array_equal(_images1, _images2, err_msg=f"Failed On {pathlib.Path(_dir).name} "
                                                                                f"on second loading")

    rmtree(tmp_path)


@DATASET
def test_tiff_stack_load_and_save(datafiles, tmp_path):
    for _dir in datafiles.listdir():
        # INGEST

        _input_image = next(pathlib.Path(_dir).glob("Video_01_of_1.tif"))
        _descriptions = next(pathlib.Path(_dir).glob("description.txt"))
        _output_folder = "".join([str(tmp_path), "\\", str(pathlib.Path(_dir).stem), "_output"])
        os.mkdir(_output_folder)
        # GET COMPARISON DESCRIPTIONS
        _descriptions = read_descriptions(_descriptions)
        # TEST
        _image1 = load_single_tiff(_input_image, _descriptions[2])
        np.testing.assert_array_equal(_image1.shape, _descriptions[2:5], err_msg=f"Failed On "
                                                                                f"{pathlib.Path(_dir).name} "
                                                                                f"on first loading")
        save_tiff_stack(_image1, _output_folder)
        _image2 = load_all_tiffs(_output_folder)
        np.testing.assert_array_equal(_image1, _image2, err_msg=f"Failed On {pathlib.Path(_dir).name} "
                                                                                f"on first loading")


@DATASET
def test_repackage_bruker_tiffs(datafiles, tmp_path):
    for _dir in datafiles.listdir():
        # INGEST
        _input_folder = next(pathlib.Path(_dir).glob("bruker_folder"))
        _descriptions = next(pathlib.Path(_dir).glob("description.txt"))
        _output_folder = "".join([str(tmp_path), "\\", str(pathlib.Path(_dir).stem), "_output"])
        # MAKE OUTPUT FOLDER
        os.mkdir(_output_folder)
        # RUN FUNCTION
        repackage_bruker_tiffs(_input_folder, _output_folder, (0, 0))
        # TEST
        _descriptions = read_descriptions(_descriptions)
        _contents = load_all_tiffs(_output_folder)
        assert _contents.shape[0] == np.cumprod(_descriptions[2])[-1], f"Failed On {pathlib.Path(_input_folder).name}"


@DATASET
def test_load_bruker_tiffs(datafiles):
    for _dir in datafiles.listdir():
        # INGEST
        _input_folder = next(pathlib.Path(_dir).glob("bruker_folder"))
        _input_image = next(pathlib.Path(_dir).glob("Video_01_of_1.tif"))
        _descriptions = next(pathlib.Path(_dir).glob("description.txt"))
        # LOAD COMPARISON
        _descriptions = read_descriptions(_descriptions)
        _image1 = load_single_tiff(_input_image, _descriptions[2])
        # TEST
        _image2 = load_bruker_tiffs(_input_folder, 1, 0)[0]
        np.testing.assert_array_equal(_image1, _image2, err_msg=f"Failed On "
                                                                                f"{pathlib.Path(_dir).name}")


@DATASET
def test_video_load_and_save(datafiles, tmp_path):
    for _dir in datafiles.listdir():
        # INGEST
        _input_file = next(pathlib.Path(_dir).glob("Video_01_of_1.tif"))
        _output_folder = "".join([str(tmp_path), "\\", str(pathlib.Path(_dir).stem), "_output"])
        _descriptions = next(pathlib.Path(_dir).glob("description.txt"))
        _descriptions = read_descriptions(_descriptions)
        # MAKE OUTPUT FOLDER
        os.mkdir(_output_folder)
        _image = load_single_tiff(_input_file, _descriptions[2])
        save_video(_image, _output_folder)
    rmtree(tmp_path)


def test_clean_up_directory(tmp_path):
    _tmp_dir_to_wipe = str(pathlib.Path(tmp_path).parent)
    rmtree(_tmp_dir_to_wipe)
