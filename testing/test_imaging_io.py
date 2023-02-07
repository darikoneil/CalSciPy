import os
import pytest
from shutil import rmtree
from src.CalSciPy.io_tools import load_all_tiffs, load_single_tiff, save_raw_binary, load_raw_binary, \
    save_single_tiff, save_tiff_stack, save_video, load_binary_meta
from src.CalSciPy.bruker import determine_bruker_folder_contents, repackage_bruker_tiffs, load_bruker_tiffs
import numpy as np
import pathlib
# noinspection PyProtectedMember
from src.CalSciPy._style import TerminalStyle


FIXTURE_DIR = "".join([os.path.abspath(os.path.join(os.getcwd(), os.pardir)), "\\testing_data"])

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
