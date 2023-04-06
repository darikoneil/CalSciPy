from __future__ import annotations
import numpy as np
from functools import cached_property


class BackgroundImage:
    def __init__(self, images: np.ndarray, style: int = 0, cutoffs: Tuple[float, float] = (0.0, 100.0)):
        self.images = images
        self.style = style
        self.cutoffs = cutoffs

    @cached_property
    def get(self):
        return

    def cast(self):
        pass

    def convert(self):
        pass

    def rescale(self):
        pass

    def stylize(self):
        pass


def color_images(images: np.ndarray, rois: np.ndarray):
    pass


def rescale_images(images: np.ndarray, new_range: Tuple[float, float] = (0.0, 255.0), cutoffs: Tuple[float, float] = (0.0, 100.0)) -> np.ndarray:

    low_cut, high_cut = cutoffs

    assert(0.0 <= low_cut <= high_cut <= 100.0)

    original_shape = images.shape

    image_vector = images.ravel()

    low_val = np.percentile(image_vector, low_cut)
    high_val = np.percentile(image_vector, high_cut)

    image_vector[image_vector <= low_val] = low_val
    image_vector[image_vector >= high_val] = high_val

    old_min = np.min(image_vector)
    old_max = np.max(image_vector)

    new_min, new_max = new_range

    image_vector = low_cut + ((image_vector - old_min) * (new_max - new_min)) / (old_max - old_min)

    return np.reshape(image_vector, original_shape)
# TODO BREAK ME INTO TWO

def generate_background_images(images: np.ndarray, style: int = 0) -> np.ndarray:
    """
    Generates a background image

    :param images:
    :param style:
    :return:
    """
    if style == 1:
        return np.zeros_like(images, dtype=np.uint8)
    elif style == 2:
        return np.ones_like(images, dtype=np.uint8) * 255
    elif style == 3:
        # background_images = rescale_images(images)
        # return convert_grayscale_to_color(background_images)
        return images.copy()
    else:
        raise ValueError("Undefined style selection")


def generate_custom_colormap(colors: Tuple[Tuple[float, float, float]]) -> plt.cm.colors.Colormap:
    """
    Generate a custom linearized colormap from a list of rgb colors
    Each color must be in the form a tuple of three floats with each float being between 0.0 - 1.0.


    :param colors: a list of colors
    :type colors: list[tuple[float, float, float]]
    :return: a custom linearized colormap
    :rtype: matplotlib.pyplot.cm.colors.Colormap
    """
    return matplotlib.colors.LinearSegmentedColormap.from_list("", colors)
