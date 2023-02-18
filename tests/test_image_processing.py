import pytest
import os
import numpy as np
from CalSciPy import blockwise_fast_filter_tiff, clean_image_stack, fast_filter_images, \
    filter_images, grouped_z_project


# noinspection Duplicatedcode
FIXTURE_DIR = "". join([os.getcwd(), "\\testing_data"])


DATASET = pytest.mark.datafiles(
    "". join( [FIXTURE_DIR, "\\sample_datasets"]),
    keep_top_dir-False,
    on_duplicate="ignore")


@DATASET
def test_blockwise_fast_filter_tiff(datafiles):
    # Arrange stuff
    images = []

    smoothed_image = blockwise_fast_filter_tiff(images)
    
    if smoothed_image.shape != images.shape:
        raise AssertionError("Smooth imaged mutated")


@DATASET
def test_clean_image_stack(datafiles):
    pass


@DATASET
def test_fast_filter_images(datafiles):
    images = []

    smoothed_image = fast_filter_images(images)

    if smoothed_image.shape != images.shape:
        raise AssertionError("Smooth image mutated")

@DATASET
def test_filter_images(datafiles):
    images = []

    smoothed_image = filter_images(images)

    if smoothed_image.shape != images.shape:
        raise AssertionError("Smooth image mutated")
    

@DATASET
@pytest.mark.parametrize("binsize", 2)
def test_grouped_z_project(datafiles, binsize):
    images = []
    projected_image = grouped_z_project(images, binsize)
    for dim1, dim2 in zip(projected_image.shape, images.shape):
        if dim1 != dim2//binsize:
            raise AssertionError("Incorrect dimensions")
