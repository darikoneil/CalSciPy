from CalSciPy.misc import generate_blocks
from CalSciPy.image_processing import _median_filter
from CalSciPy.io_tools import load_binary

import numpy as np
import cupy
import cupyx.scipy.ndimage

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib import pyplot as plt


binary_path = "D:\\EM_B\\fov_reference\\results"

mask = np.ones((7, 3, 3))

images = load_binary(binary_path)

filtered_images = images.copy()

frames = filtered_images.shape[0]

block_size = 1000

block_buffer = 100

block_gen = generate_blocks(range(frames), block_size, block_buffer)

block1 = next(block_gen)
block2 = next(block_gen)
block3 = next(block_gen)
block4 = next(block_gen)

ref_image = cupyx.scipy.ndimage.median_filter(cupy.asarray(filtered_images), footprint=mask).get()

s1 = cupyx.scipy.ndimage.median_filter(cupy.asarray(filtered_images[block1, :, :]), footprint=mask).get()

np.testing.assert_equal(s1[0:900, :, :], ref_image[0:900, :, :])
