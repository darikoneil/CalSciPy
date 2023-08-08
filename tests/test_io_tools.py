import pytest
from shutil import rmtree
from pathlib import Path
import numpy as np
from tests.conftest import SAMPLES_DATASETS_DIRECTORY
from tests.helpers import read_descriptions
# noinspection PyProtectedMember
from CalSciPy.io_tools import load_images, _load_single_tif, _load_many_tif, save_images, _save_single_tif, \
    _save_many_tif, load_binary, save_binary


DATASET = pytest.mark.datafiles(
    SAMPLES_DATASETS_DIRECTORY,
    keep_topdirectory=False,
    on_duplicate="ignore",
)


@DATASET
def test_single_page_tifs(datafiles, tmp_path, matrix):
    for directory in datafiles.listdir():
        # Description of Expected
        descriptions = Path(directory).joinpath("description.txt")
        descriptions = read_descriptions(descriptions)

        # single image
        single_image_file = Path(directory).joinpath("single_page", "images.tif")

        # output folder
        output_folder = Path(tmp_path).joinpath("".join([Path(directory).stem, "_output"]))

        # test single image loading
        image_abstracted_method = load_images(single_image_file)
        image_implement_method = _load_single_tif(single_image_file)
        np.testing.assert_array_equal(image_abstracted_method.shape, [1, *descriptions[3:5]],
                                      err_msg=f"Mismatch: failed on dataset {Path(directory).name} "
                                              f"during first loading using abstracted")
        np.testing.assert_array_equal(image_implement_method.shape, [1, *descriptions[3:5]],
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
        # test 2D single image
        save_images(output_folder, image_abstracted_method.reshape(image_abstracted_method.shape[1:]))
        images_abstracted_2d = load_images(output_folder)
        np.testing.assert_array_equal(image_abstracted_method_reloaded, image_implement_method_reloaded,
                                      err_msg=f"Image Mismatch between loading with implementation "
                                              f"and abstracted methods")
        np.testing.assert_array_equal(image_abstracted_method_reloaded, image_abstracted_method,
                                      err_msg=f"Image mismatch between saved and loaded image")
        np.testing.assert_array_equal(
            image_abstracted_method_reloaded,
            images_abstracted_2d,
            err_msg=f"Image mismatch between 3D & 2D image")
        # test exact filename
        output_file = Path(tmp_path).joinpath("exact_filename.tif")
        save_images(output_file, image_abstracted_method)
        image_reloaded = load_images(output_file)

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
        descriptions = Path(directory).joinpath("description.txt")
        descriptions = read_descriptions(descriptions)

        # single image
        imaging_file = Path(directory).joinpath("multi_page", "images.tif")

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

        # test exact filename
        output_file = Path(tmp_path).joinpath("exact_filename.tif")
        save_images(output_file, image_abstracted_method)
        image_reloaded = load_images(output_file)

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
        descriptions = Path(directory).joinpath("description.txt")
        descriptions = read_descriptions(descriptions)

        # single image
        imaging_folder = Path(directory).joinpath("many")

        # output folder
        output_folder_0 = Path(tmp_path).joinpath("".join([Path(directory).stem, "_output_0"]))
        Path.mkdir(output_folder_0, parents=True, exist_ok=True)
        output_folder_1 = Path(tmp_path).joinpath("".join([Path(directory).stem, "_output_1"]))
        Path.mkdir(output_folder_1, parents=True, exist_ok=True)

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
        save_images(output_folder_0, image_abstracted_method, size_cap=0.01)
        _save_many_tif(output_folder_1.joinpath("images"), image_abstracted_method, size_cap=0.01)
        image_abstracted_method_reloaded = load_images(output_folder_0)
        image_implement_method_reloaded = load_images(output_folder_1)
        np.testing.assert_array_equal(image_abstracted_method_reloaded, image_implement_method_reloaded,
                                      err_msg=f"Image Mismatch between loading with implementation "
                                              f"and abstracted methods")
        np.testing.assert_array_equal(image_abstracted_method_reloaded, image_abstracted_method,
                                      err_msg=f"Image mismatch between saved and loaded image")

        # test exact filename
        output_file = Path(tmp_path).joinpath("exact_filename.tif")
        save_images(output_file, image_abstracted_method)

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
def test_binary(datafiles, tmp_path, matrix):
    for directory in datafiles.listdir():
        # Description of Expected
        descriptions = Path(directory).joinpath("description.txt")
        descriptions = read_descriptions(descriptions)
        frames, y, x = descriptions[2:]

        # single image
        imaging_folder = Path(directory).joinpath("binary")

        # output folder
        output_folder = Path(tmp_path).joinpath("".join([Path(directory).stem, "_output_0"]))

        images = load_binary(imaging_folder)

        np.testing.assert_array_equal(images.shape[1:], descriptions[3:5],
                                      err_msg=f"Mismatch: failed on dataset {Path(directory).name} "
                                              f"during first loading using abstracted")

        save_binary(output_folder, images)

        images_reloaded = load_binary(output_folder)

        np.testing.assert_array_equal(images, images_reloaded,
                                      err_msg=f"Mismatch: failed matching original and reloaded dataset")

        # make sure memory map loads
        image_memory_mapped = load_binary(output_folder, mapped=True)

        # make sure we can save & load a non-default filename
        exact_output_folder = Path(tmp_path).joinpath("exact_folder")
        save_binary(exact_output_folder, images, name="exact_filename")
        images_reloaded_exact_filename = load_binary(exact_output_folder.joinpath("exact_filename"))

        # make sure we can load with interpolated metadata
        interp_images = load_binary(exact_output_folder.joinpath("exact_filename"),
                                    missing_metadata={"dtype": np.uint16, "y": y, "x": x})

        np.testing.assert_array_equal(images_reloaded_exact_filename, interp_images,
                                      err_msg=f"Mismatch: failed matching original and reloaded dataset")

        # we also fail if missing sufficient metadata
        with pytest.raises(AttributeError):
            load_binary(exact_output_folder.joinpath("exact_filename"),
                        missing_metadata={"frames": frames, "y": y, "x": x})
        with pytest.raises(AttributeError):
            load_binary(exact_output_folder.joinpath("exact_filename"),
                        missing_metadata={"dtype": np.uint16, "x": x})

    # generic test exceptions
    with pytest.raises(FileNotFoundError):
        load_binary(Path.cwd())
    with pytest.raises(ValueError):
        load_binary("C:\\&^6* ***%")  # FAIL PERMITTED CHARS
    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        load_binary(125.6)  # fail with bad type so we don't do anything unexpected
    with pytest.raises(ValueError):
        save_binary("C:\\&^6*", matrix)  # FAIL VALIDATE PATH PERMITTED CHARS
    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        save_binary(125.6, matrix)  # fail with bad type so we don't do anything unexpected

    # the rest of failures are fine being trial by forgiveness
