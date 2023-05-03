from tests.conftest import SAMPLES_VARIABLES_DIRECTORY
from CalSciPy.coloring import rescale_images, cutoff_images
import numpy as np
from pathlib import Path


path = SAMPLES_VARIABLES_DIRECTORY.joinpath("sample_image.npy")

image = np.load(path)

images = image.copy()
new_range = (0.0, 255.0)
in_place = True

image_vector = images.ravel()
old_min = np.min(image_vector)
old_max = np.max(image_vector)
new_min, new_max = new_range


image_vector = new_min + ((image_vector - old_min) * (new_max - new_min)) / (old_max - old_min)

f = np.reshape(image_vector, images.shape)


