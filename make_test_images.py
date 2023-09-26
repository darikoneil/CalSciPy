from CalSciPy.optics.holo import SLM
from CalSciPy.roi_tools import calculate_mask
from slmsuite.holography.algorithms import Hologram
import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib import pyplot as plt
from PIL import Image

slm_size = (512, 512)

targets = [calculate_mask(centroid=(target[0], target[1]), radii=10, reference_shape=(512, 512)) for target in
           zip((128, 255, 383), (128, 255, 255))]

target = np.sum(targets, axis=0)

fig, ax = plt.subplots(1, 1)
ax.imshow(target)

DEFAULT_SDK_PATH = "C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus\\"
DEFAULT_LUT_PATH = "C:\\Users\\rylab\\Desktop\\slm4317_at1064_P8.lut"

slm = SLM(brand="meadowlark", hdmi=False, sdk_path=DEFAULT_SDK_PATH, lut_path=DEFAULT_LUT_PATH)

hologram = Hologram(target, slm_shape=(512, 512))

hologram.optimize(method="GS", maxiter=100)

hologram.plot_nearfield(padded=False, cbar=True)

hologram.plot_farfield(cbar=True, units="knm")

true_phase = slm._slm._phase2gray(hologram.extract_phase())

true_phase = Image.fromarray(true_phase)

true_phase.save("test_slide_one.bmp")

flipped_true_phase = slm._slm._phase2gray(hologram.extract_phase())

flipped_true_phase -= 255

flipped_true_phase = np.absolute(flipped_true_phase)

flipped_true_phase = Image.fromarray(flipped_true_phase)

flipped_true_phase.save("test_slide_two.bmp")
