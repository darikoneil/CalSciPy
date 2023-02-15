import os
import pytest
from shutil import rmtree
from CalSciPy.io_tools import load_all_tiffs, load_single_tiff, save_raw_binary, load_raw_binary, \
    save_single_tiff, save_tiff_stack, save_video, load_binary_meta
from CalSciPy.bruker import determine_bruker_folder_contents, repackage_bruker_tiffs, load_bruker_tiffs
import numpy as np
import pathlib
# noinspection PyProtectedMember
from PPVD.style import TerminalStyle

# noinspection DuplicatedCode
FIXTURE_DIR = "".join([os.getcwd(), "\\testing_data"])

DATASET = pytest.mark.datafiles(
    "".join([FIXTURE_DIR, "\\sample_datasets"]),
    keep_top_dir=False,
    on_duplicate="ignore",
)


def read_descriptions(file):
    return np.genfromtxt(str(file), delimiter=",", dtype="int")


@DATASET
def test_determine_bruker_folder_contents_passes(datafiles):
    for _dir in datafiles.listdir():

        _input_folder = next(pathlib.Path(_dir).glob("bruker_folder"))
        _descriptions = next(pathlib.Path(_dir).glob("description.txt"))
        _descriptions = read_descriptions(_descriptions)

        _contents = determine_bruker_folder_contents(_input_folder)

        for _test in enumerate(["Channel", "Plane", "Frame", "Height", "Width"]):
            assert _contents[_test[0]] == _descriptions[_test[0]], f"{TerminalStyle.GREEN}Description Mismatch: " \
                                                                   f"{TerminalStyle.YELLOW} failed on dataset " \
                                                                   f"{TerminalStyle.BLUE}{pathlib.Path(_dir).name}: " \
                                                                   f"{TerminalStyle.ORANGE}{_test[1]}" \
                                                                   f"{TerminalStyle.YELLOW} detection " \
                                                                   f"{TerminalStyle.RESET}"
        return

    rmtree(datafiles)


def test_determine_bruker_folder_contents_fails():
    with pytest.raises(ValueError):
        determine_bruker_folder_contents("")  # FAIL VALIDATE PATH NO ROOT/DRIVE
    with pytest.raises(ValueError):
        determine_bruker_folder_contents("C:\\&^6*")  # FAIL VALIDATE PATH PERMITTED CHARS
    with pytest.raises(FileNotFoundError):
        determine_bruker_folder_contents("C:\\673469346349673496734967349673205-3258-32856-3486")
        # FAIL VALIDATE PATH NOT FOUND
    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        determine_bruker_folder_contents(125.6)
        # FAIL WRONG TYPE


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
        assert _contents.shape[0] == np.cumprod(_descriptions[2])[-1], f"{TerminalStyle.GREEN}Image Mismatch: " \
                                                                       f"{TerminalStyle.YELLOW}failed on dataset" \
                                                                       f"{TerminalStyle.BLUE} " \
                                                                       f"{pathlib.Path(_input_folder).name} " \
                                                                       f"{TerminalStyle.RESET}"


def test_repackage_bruker_tiffs_fails():
    with pytest.raises(ValueError):
        repackage_bruker_tiffs("C:\\&^6", "C:\\&^6")  # FAIL PATH CHARACTERS
    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        repackage_bruker_tiffs("C:\\", 125.6)  # FAIL PATH TYPE
    with pytest.raises(FileNotFoundError):
        repackage_bruker_tiffs("C:\\1381945328953298532895", "C:\\")  # FAIL PATH EXISTS


@DATASET
def test_load_bruker_tiffs(datafiles):
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
        np.testing.assert_array_equal(_image1, _image2, err_msg=f"{TerminalStyle.GREEN}Image Mismatch: "
                                                                f"{TerminalStyle.YELLOW}failed on dataset"
                                                                f"{TerminalStyle.BLUE} "
                                                                f"{pathlib.Path(_input_folder).name} "
                                                                f"{TerminalStyle.RESET}")


def test_load_bruker_tiff_fails():
    with pytest.raises(ValueError):
        load_bruker_tiffs("")  # FAIL VALIDATE PATH NO ROOT/DRIVE
    with pytest.raises(ValueError):
        load_bruker_tiffs("C:\\&^6*")  # FAIL VALIDATE PATH PERMITTED CHARS
    with pytest.raises(FileNotFoundError):
        load_bruker_tiffs("C:\\673469346349673496734967349673205-3258-32856-3486")
        # FAIL VALIDATE PATH NOT FOUND
    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        load_bruker_tiffs(125.6)
        # FAIL WRONG TYPE
