import pytest
from shutil import rmtree
from pathlib import Path
import numpy as np

from tests.conftest import SAMPLES_DATASETS_DIRECTORY
from tests.helpers import read_descriptions
# noinspection PyProtectedMember
from CalSciPy.io_tools import load_images, _load_single_tif, _load_many_tif, save_images, _save_single_tif, \
    _save_many_tif


DATASET = pytest.mark.datafiles(
    SAMPLES_DATASETS_DIRECTORY,
    keep_topdirectory=False,
    on_duplicate="ignore",
)


@DATASET
def test_single_page_tifs(datafiles, tmp_path, matrix):
    for directory in datafiles.listdir():
        # Description of Expected
        descriptions = next(Path(directory).glob("description.txt"))
        descriptions = read_descriptions(descriptions)

        # single image
        single_image_file = next(Path(directory).glob("single.tif"))

        # output folder
        output_folder = Path(tmp_path).joinpath("".join([Path(directory).stem, "_output"]))

        # test single image loading
        image_abstracted_method = load_images(single_image_file)
        image_implement_method = _load_single_tif(single_image_file)
        np.testing.assert_array_equal(image_abstracted_method.shape, descriptions[3:5],
                                      err_msg=f"Mismatch: failed on dataset {Path(directory).name} "
                                              f"during first loading using abstracted")
        np.testing.assert_array_equal(image_implement_method.shape, descriptions[3:5],
                                      err_msg=f"Mismatch: failed on dataset {Path(directory).name} "
                                              f"during first loading using implementation")
        np.testing.assert_array_equal(image_implement_method, image_abstracted_method, err_msg=f"Image Mismatch: failed"
                                                                                               f"match between"
                                                                                               f"implementation and"
                                                                                               f"abstracted methods")
        # test single image saving
        save_images(output_folder, image_abstracted_method)
        _save_single_tif(output_folder, image_abstracted_method)
        image_abstracted_method_reloaded = load_images(output_folder)
        image_implement_method_reloaded = load_images(output_folder)
        np.testing.assert_array_equal(image_abstracted_method_reloaded, image_implement_method_reloaded,
                                      err_msg=f"Image Mismatch between loading with implementation "
                                              f"and abstracted methods")
        np.testing.assert_array_equal(image_abstracted_method_reloaded, image_abstracted_method,
                                      err_msg=f"Image mismatch between saved and loaded image")

    # test exceptions
    with pytest.raises(ValueError):
        load_images("C:\\&^6* ***%")  # FAIL PERMITTED CHARS
    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        load_images(125.6)  # fail with bad type so we don't do anything unexpected
    with pytest.raises(ValueError):
        save_images("C:\\&^6*", matrix)  # FAIL VALIDATE PATH PERMITTED CHARS
    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        save_images(125.6, matrix)  # fail with bad type so we don't do anything unexpected
    # the rest of failures are fine being trial by forgiveness

    rmtree(tmp_path)


@DATASET
def test_multi_page_tifs(datafiles, tmp_path, matrix):
    for directory in datafiles.listdir():
        # Description of Expected
        descriptions = next(Path(directory).glob("description.txt"))
        descriptions = read_descriptions(descriptions)

        # single image
        imaging_file = next(Path(directory).glob("Video_01_of_1.tif"))

        # output folder
        output_folder = Path(tmp_path).joinpath("".join([Path(directory).stem, "_output"]))

        # test single image loading
        image_abstracted_method = load_images(imaging_file)
        image_implement_method = _load_single_tif(imaging_file)
        np.testing.assert_array_equal(image_abstracted_method.shape[1:], descriptions[3:5],
                                      err_msg=f"Mismatch: failed on dataset {Path(directory).name} "
                                              f"during first loading using abstracted")
        np.testing.assert_array_equal(image_implement_method.shape[1:], descriptions[3:5],
                                      err_msg=f"Mismatch: failed on dataset {Path(directory).name} "
                                              f"during first loading using implementation")
        np.testing.assert_array_equal(image_implement_method, image_abstracted_method, err_msg=f"Image Mismatch: failed"
                                                                                               f"match between"
                                                                                               f"implementation and"
                                                                                               f"abstracted methods")
        # test single image saving
        save_images(output_folder, image_abstracted_method)
        _save_single_tif(output_folder, image_abstracted_method)
        image_abstracted_method_reloaded = load_images(output_folder)
        image_implement_method_reloaded = load_images(output_folder)
        np.testing.assert_array_equal(image_abstracted_method_reloaded, image_implement_method_reloaded,
                                      err_msg=f"Image Mismatch between loading with implementation "
                                              f"and abstracted methods")
        np.testing.assert_array_equal(image_abstracted_method_reloaded, image_abstracted_method,
                                      err_msg=f"Image mismatch between saved and loaded image")

    # test exceptions
    with pytest.raises(ValueError):
        load_images("C:\\&^6* ***%")  # FAIL PERMITTED CHARS
    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        load_images(125.6)  # fail with bad type so we don't do anything unexpected
    with pytest.raises(ValueError):
        save_images("C:\\&^6*", matrix)  # FAIL VALIDATE PATH PERMITTED CHARS
    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        save_images(125.6, matrix)  # fail with bad type so we don't do anything unexpected
    # the rest of failures are fine being trial by forgiveness

    rmtree(tmp_path)


@DATASET
def test_many_tifs(datafiles, tmp_path, matrix):
    for directory in datafiles.listdir():
        # Description of Expected
        descriptions = next(Path(directory).glob("description.txt"))
        descriptions = read_descriptions(descriptions)

        # single image
        imaging_folder = Path(directory).joinpath("many_tif")

        # output folder
        output_folder_0 = Path(tmp_path).joinpath("".join([Path(directory).stem, "_output_0"]))
        output_folder_1 = Path(tmp_path).joinpath("".join([Path(directory).stem, "_output_1"]))

        # test single image loading
        image_abstracted_method = load_images(imaging_folder)
        image_implement_method = _load_many_tif(imaging_folder)
        np.testing.assert_array_equal(image_abstracted_method.shape[1:], descriptions[3:5],
                                      err_msg=f"Mismatch: failed on dataset {Path(directory).name} "
                                              f"during first loading using abstracted")
        np.testing.assert_array_equal(image_implement_method.shape[1:], descriptions[3:5],
                                      err_msg=f"Mismatch: failed on dataset {Path(directory).name} "
                                              f"during first loading using implementation")
        np.testing.assert_array_equal(image_implement_method, image_abstracted_method,
                                      err_msg=f"Image Mismatch: failed"
                                              f"match between"
                                              f"implementation and"
                                              f"abstracted methods")
        # test single image saving
        save_images(output_folder_0, image_abstracted_method, size_cap=0.06)
        _save_many_tif(output_folder_1, image_abstracted_method, size_cap=0.06)
        image_abstracted_method_reloaded = load_images(output_folder_0)
        image_implement_method_reloaded = load_images(output_folder_1)
        np.testing.assert_array_equal(image_abstracted_method_reloaded, image_implement_method_reloaded,
                                      err_msg=f"Image Mismatch between loading with implementation "
                                              f"and abstracted methods")
        np.testing.assert_array_equal(image_abstracted_method_reloaded, image_abstracted_method,
                                      err_msg=f"Image mismatch between saved and loaded image")

    # test exceptions
    with pytest.raises(ValueError):
        load_images("C:\\&^6* ***%")  # FAIL PERMITTED CHARS
    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        load_images(125.6)  # fail with bad type so we don't do anything unexpected
    with pytest.raises(ValueError):
        save_images("C:\\&^6*", matrix)  # FAIL VALIDATE PATH PERMITTED CHARS
    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        save_images(125.6, matrix)  # fail with bad type so we don't do anything unexpected
    # the rest of failures are fine being trial by forgiveness

    rmtree(tmp_path)

