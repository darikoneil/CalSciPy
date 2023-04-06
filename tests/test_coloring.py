import pytest
from CalSciPy.coloring import rescale_images


def test_rescale_images(simple_matrix):
    original = simple_matrix.copy()
    m = rescale_images(simple_matrix, (0.0, 255.0), (0.0, 50.0))
    print(f"{original=}")
    print(f"{m=}")
