from __future__ import annotations

import numpy as np

from scipy.ndimage.filters import median_filter
from scipy.signal import convolve
from scipy.ndimage import affine_transform
from scipy.optimize import least_squares

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.backends.qt_compat import QtCore, QtWidgets, QtGui
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.gridspec import GridSpec


class PSF:
    def __init__(self, psf, headless=False):
        self.scaling = (0.5, 0.1, 0.1)
        self.planes, self.y_pixels, self.x_pixels = psf.shape
        self.image = psf.astype(np.float64)
        self.denoised = None
        self.FWHM = None
        self.med_filt = 1
        self.z_fit_order = 1
        self.z_max = None

        self._denoise()

        # if not headless:
        #    interactive_psf(self)

    def _denoise(self):
        # median filter
        self.denoised = median_filter(self.image, footprint=np.ones((1, 1, 1)), mode="mirror")

        # subtract baseline
        self.denoised -= np.median(self.denoised)

        # bicubic interpolate downward
        trans_matrix = np.zeros((4, 4))
        trans_matrix[0, 0] = 1
        trans_matrix[1, 1] = 4
        trans_matrix[2, 2] = 4
        trans_matrix[3, 3] = 1

        output_shape = (self.planes,
                        int(self.y_pixels // 4),
                        int(self.x_pixels // 4))

        denoised = np.ones(output_shape, dtype=np.float64) * 4

        self.denoised = affine_transform(self.denoised,
                                         trans_matrix,
                                         output=denoised,
                                         output_shape=output_shape)

        for plane in range(output_shape[0]):
            self.denoised[plane, :, :] = convolve(self.denoised[plane, :, :],
                                                  np.ones((2, 2)),
                                                  mode="same",
                                                  method="direct")

    @property
    def center(self):
        return np.argmax(np.max(self.denoised, axis=(1, 2)))


def interactive_psf(psf):
    fig = plt.figure()
    gs = GridSpec(2, 2)
    # current plane
    ax1 = plt.subplot(gs[0, 0])
    # center plane
    ax2 = plt.subplot(gs[0, 1])
    # XZ
    ax3 = plt.subplot(gs[1, 0])
    # YZ
    ax4 = plt.subplot(gs[1, 1])

    current_plane = 0

    # current plane
    ax1.imshow(psf.denoised[current_plane, :, :], cmap="mako")
    ax1.set_title("Current Plane")
    ax1.set_xticks([])
    ax1.set_yticks([])

    # center plane
    ax2.imshow(psf.denoised[psf.center, :, :], cmap="mako")
    ax2.set_title("Center Plane")
    ax2.set_xticks([])
    ax2.set_yticks([])

    ax3.set_title("XZ")
    ax3.set_xticks([])

    ax4.set_title("YZ")
    ax4.set_xticks([])

    def update():
        cp = plane_slider.value()
        ax1.clear()
        ax1.imshow(psf.denoised[cp, :, :], cmap="mako")
        fig.canvas.draw()

    plane_slider = QtWidgets.QSlider(QtCore.Qt.Vertical)
    plane_slider.setRange(0, 50)
    plane_slider.setSingleStep(1)
    plane_slider.setValue(0)
    plane_slider.sliderMoved.connect(update)
    vbox = QtWidgets.QVBoxLayout()
    vbox.addWidget(plane_slider)
    fig.canvas.setLayout(vbox)
    plt.show()

    return fig


def identify_peak(intensity, pixels, units):

    x = np.arange(pixels) * units
    y = intensity


def single_term_gaussian(theta, x):
    c0, c1, c2 = theta
    numerator = (x - c1)
    denominator = c2
    exponent = (numerator / denominator)**2
    exponent *= -1
    return c0 * np.exp(exponent)


psf = PSF(np.load("C:\\Users\\YUSTE\\Desktop\\PSF.npy"))


fig, ax = plt.subplots(1, 1)
ax.imshow(psf.denoised[psf.center, :, :])


_ = interactive_psf(psf)
