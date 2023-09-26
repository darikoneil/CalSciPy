from CalSciPy.optics.holo import SLM
from slmsuite.holography.toolbox.phase import blaze, lens
import numpy as np


import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib import pyplot as plt
import seaborn as sns

from slmsuite.holography.algorithms import Hologram
from slmsuite.hardware.slms.slm import SLM
from slmsuite.hardware.cameras.camera import Camera
from slmsuite.hardware.cameraslms import FourierSLM
from slmsuite.holography import toolbox
from slmsuite.holography.algorithms import SpotHologram


roi = np.load(".\\tests\\testing_samples\\variables\\sample_rois.npy", allow_pickle=True).item()
target = np.zeros((512, 512))

for r in roi.values():
    ypix = r.get("ypix")
    xpix = r.get("xpix")
    for y, x in zip(ypix.tolist(), xpix.tolist()):
        target[y, x] = 1

slm_size = (512, 512)

hologram = Hologram(target, slm_shape=slm_size)

zoombox = hologram.plot_farfield(source=hologram.target, cbar=True)

slm = SLM(512, 512, dx_um=15, dy_um=15)
camera = Camera(512, 512)
setup = FourierSLM(camera, slm)
hologram.cameraslm = setup

fs = FourierSLM(camera, slm)

xlist = np.arange(100, 512, 100)
xgrid, ygrid = np.meshgrid(xlist, xlist)
square = np.vstack((xgrid.ravel(), ygrid.ravel()))
hologram = SpotHologram(shape=(512, 512), spot_vectors=square, basis='knm')
hologram.plot_farfield(hologram.target)


lg90_phase = toolbox.phase.laguerre_gaussian(
    slm,
    l=9,                                        # A larger azimuthal wavenumber
    p=0
)


for units in ["kxy", "knm", "freq", "deg"]:
    hologram.plot_farfield(source=target, units=units, figsize=(8, 4))

#hologram.optimize(method='GS', maxiter=5)

#hologram.plot_nearfield(cbar=True)
#hologram.plot_farfield(limits=zoombox, cbar=True, title='FF Amp')

#hologram.shape = hologram.calculate_padded_shape(setup, precision=0.5/(slm.dx*slm.shape[0]))

#target_padded = toolbox.pad(target, hologram.shape)

# hologram = Hologram(target_padded, slm_shape=slm.shape)
# zoombox_padded = hologram.plot_farfield(target_padded)
# hologram.optimize(method='GS', maxiter=5)

#hologram.plot_nearfield(padded=True, cbar=True)
#hologram.plot_farfield(cbar=True, units="deg", title='FF Amp', limits=zoombox_padded)

slm.set_measured_amplitude_analytic(8, units="mm")

hologram_gauss = Hologram(target, slm_shape=slm.shape, amp=slm.measured_amplitude)
hologram_gauss.optimize(method='WGS-Leonardo', maxiter=20)

hologram_gauss.plot_nearfield(padded=False, cbar=True)
hologram_gauss.plot_farfield(cbar=True, limits=zoombox, units="kxy")


# Instead of picking a few points, make a rectangular grid in the knm basis
array_holo = SpotHologram.make_rectangular_array(
    slm_size,
    array_shape=(10, 10),
    array_pitch=(50, 50),
    basis='knm'
)
zoom = array_holo.plot_farfield(source=array_holo.target, title='Initialized Nearfield')


# ij is image pixels

# kxy is Normalized (floating point) basis of the SLM’s
# k-space in normalized units. Centered at (kx, ky) = (0, 0). This basis is what the SLM projects in angular space (which maps to the camera’s image space via the Fourier transform implemented by free space and solidified by a lens).

#knm
# Pixel basis of the SLM’s computational
#-space. Centered at (kn, km) = (0, 0). "knm" is a discrete version of the continuous "kxy". This is important because holograms need to be stored in computer memory, a discrete medium with pixels, rather than being purely continuous. For instance, in SpotHologram, spots targeting specific continuous angles are rounded to the nearest discrete pixels of "knm" space in practice. Then, this "knm" space image is handled as a standard image/array, and operations such as the discrete Fourier transform (instrumental for numerical hologram optimization) can be applied.
