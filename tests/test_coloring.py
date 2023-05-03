import pytest
from CalSciPy.coloring import cutoff_images, rescale_images
import numpy as np


@pytest.mark.parametrize("in_place", [True, False])
def test_cutoff_images(sample_image, cutoff_reference, in_place):

    changed = cutoff_images(sample_image, (25.0, 75.0), in_place)

    np.testing.assert_equal(changed, cutoff_reference)

    if in_place:
        np.testing.assert_equal(changed, sample_image)
    else:
        assert not np.array_equal(changed, sample_image)


@pytest.mark.parametrize("in_place", [True, False])
def test_rescale_images(sample_image, rescale_reference, in_place):

    changed = rescale_images(sample_image, in_place=in_place)

    np.testing.assert_equal(changed, rescale_reference)

    if in_place:
        np.testing.assert_equal(changed, sample_image)
    else:
        assert not np.array_equal(changed, sample_image)
