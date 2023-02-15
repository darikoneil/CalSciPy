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
        _image = load_single_tiff(_input_image)
        np.testing.assert_array_equal(_image.shape, [1, *_descriptions[3:5]], err_msg=f"{TerminalStyle.GREEN}"
                                                                                      f"Image Mismatch: "
                                                                                      f"{TerminalStyle.YELLOW}"
                                                                                      f"failed on dataset " 
                                                                                      f"{TerminalStyle.BLUE}"
                                                                                      f"{pathlib.Path(_dir).name}"
                                                                                      f"{TerminalStyle.YELLOW}"
                                                                                      f" during first loading"
                                                                                      f"{TerminalStyle.RESET}")
        save_single_tiff(_image, _output_file)
        _image2 = load_single_tiff(_output_file)
        np.testing.assert_array_equal(_image, _image2, err_msg=f"{TerminalStyle.GREEN}Image Mismatch: "
                                                               f"{TerminalStyle.YELLOW}failed on dataset " 
                                                               f"{TerminalStyle.BLUE}{pathlib.Path(_dir).name}" 
                                                               f"{TerminalStyle.YELLOW} during second loading"
                                                               f"{TerminalStyle.RESET}")

    rmtree(tmp_path)


def test_single_tiff_fails():
    # LOAD
    with pytest.raises(ValueError):
        load_single_tiff("")  # FAIL VALIDATE PATH NO ROOT/DRIVE
    with pytest.raises(ValueError):
        load_single_tiff("C:\\&^6*")  # FAIL VALIDATE PATH PERMITTED CHARS
    with pytest.raises(FileNotFoundError):
        load_single_tiff("C:\\673469346349673496734967349673205-3258-32856-3486")
        # FAIL VALIDATE PATH NOT FOUND
    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        load_single_tiff(125.6)
    # SAVE
    with pytest.raises(ValueError):
        save_single_tiff([1, 2, 3, 4, 5], "")  # FAIL VALIDATE PATH NO ROOT/DRIVE
    with pytest.raises(ValueError):
        save_single_tiff([1, 2, 3, 4, 5], "C:\\&^6*")  # FAIL VALIDATE PATH PERMITTED CHARS
    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        save_single_tiff([1, 2, 3, 4, 5], 125.6)  # FAIL TYPE
    with pytest.raises(ValueError):
        save_single_tiff([1, 2, 3, 4, 5], "C:\\file.mp4")  # FAIL EXTENSION


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
        _images1 = load_single_tiff(_input_image)
        np.testing.assert_array_equal(_images1.shape, _descriptions[2:5], err_msg=f"{TerminalStyle.GREEN} "
                                                                                  f"Image Mismatch: "
                                                                                  f"{TerminalStyle.YELLOW}"
                                                                                  f"failed on dataset "
                                                                                  f"{TerminalStyle.BLUE}"
                                                                                  f"{pathlib.Path(_dir).name} "
                                                                                  f"{TerminalStyle.YELLOW}"
                                                                                  f"during first loading"
                                                                                  f"{TerminalStyle.RESET}")
        save_raw_binary(_images1, _output_folder)
        _images2 = load_raw_binary(_output_folder)
        np.testing.assert_array_equal(_images1, _images2, err_msg=f"{TerminalStyle.GREEN}Image Mismatch: "
                                                                  f"{TerminalStyle.YELLOW}failed on dataset "
                                                                  f"{TerminalStyle.BLUE}{pathlib.Path(_dir).name} "
                                                                  f"{TerminalStyle.YELLOW} during second loading "
                                                                  f"{TerminalStyle.RESET}")

    rmtree(tmp_path)


def test_binary_load_meta_fails():
    with pytest.raises(ValueError):
        load_binary_meta("")  # FAIL VALIDATE PATH NO ROOT/DRIVE
    with pytest.raises(ValueError):
        load_binary_meta("C:\\&^6*")  # FAIL VALIDATE PATH PERMITTED CHARS
    with pytest.raises(FileNotFoundError):
        load_binary_meta("C:\\673469346349673496734967349673205-3258-32856-3486")
        # FAIL VALIDATE PATH NOT FOUND
    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        load_binary_meta(125.6)
        # FAIL WRONG TYPE
    with pytest.raises(FileNotFoundError):
        load_binary_meta("C:\\file")  # No extension but adds then fails
    with pytest.raises(ValueError):
        load_binary_meta("C:\\file.mp4")  # No extension but catches


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
        _image1 = load_single_tiff(_input_image)
        np.testing.assert_array_equal(_image1.shape, _descriptions[2:5], err_msg=f"{TerminalStyle.GREEN}"
                                                                                 f"Image Mismatch: "
                                                                                 f"{TerminalStyle.YELLOW}"
                                                                                 f"failed on dataset "
                                                                                 f"{TerminalStyle.BLUE}"
                                                                                 f"{pathlib.Path(_dir).name} "
                                                                                 f"{TerminalStyle.YELLOW}"
                                                                                 f"during first loading "
                                                                                 f"{TerminalStyle.RESET}")
        save_tiff_stack(_image1, _output_folder)
        _image2 = load_all_tiffs(_output_folder)
        np.testing.assert_array_equal(_image1, _image2, err_msg=f"{TerminalStyle.GREEN}"
                                                                f"Image Mismatch: "
                                                                f"{TerminalStyle.YELLOW}"
                                                                f"failed on dataset "
                                                                f"{TerminalStyle.BLUE}"
                                                                f"{pathlib.Path(_dir).name} "
                                                                f"{TerminalStyle.YELLOW}"
                                                                f"during second loading "
                                                                f"{TerminalStyle.RESET}")


def test_tiff_stacks_fails():
    # LOAD
    with pytest.raises(ValueError):
        load_all_tiffs("")  # FAIL VALIDATE PATH NO ROOT/DRIVE
    with pytest.raises(ValueError):
        load_all_tiffs("C:\\&^6*")  # FAIL VALIDATE PATH PERMITTED CHARS
    with pytest.raises(FileNotFoundError):
        load_all_tiffs("C:\\673469346349673496734967349673205-3258-32856-3486")
        # FAIL VALIDATE PATH NOT FOUND
    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        load_all_tiffs(125.6)
        # FAIL WRONG TYPE
    # SAVE
    with pytest.raises(ValueError):
        save_tiff_stack([1, 2, 3, 4, 5], "")  # FAIL VALIDATE PATH NO ROOT/DRIVE
    with pytest.raises(ValueError):
        save_tiff_stack([1, 2, 3, 4, 5], "C:\\&^6*")  # FAIL VALIDATE PATH PERMITTED CHARS
    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        save_tiff_stack([1, 2, 3, 4, 5], 125.6)  # FAIL TYPE


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
        _image = load_single_tiff(_input_file)
        save_video(_image, _output_folder)
    rmtree(tmp_path)


def test_clean_up_directory(tmp_path):
    _tmp_dir_to_wipe = str(pathlib.Path(tmp_path).parent)
    rmtree(_tmp_dir_to_wipe)
