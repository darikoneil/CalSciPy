from __future__ import annotations
from functools import cached_property
import numpy as np

from scipy.signal import convolve
from scipy.ndimage import median_filter, affine_transform
from scipy.optimize import minimize, least_squares

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.backends.qt_compat import QtCore, QtWidgets, QtGui
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.gridspec import GridSpec


class PSF:
    # noinspection PyShadowingNames
    def __init__(self,
                 stack,
                 filter_shape=(1, 1, 1),
                 scaling=(1, 1, 1)
                 ):
        # actual observed values
        self.stack = stack.astype(np.float64)
        # denoised values
        self.denoised = None
        # filter for denoising
        self.filter_shape = filter_shape
        # units
        self.scaling = scaling
        # dimensions
        self.planes, self.y_pixels, self.x_pixels = stack.shape

        # conduct denoising
        self._denoise()

    def _denoise(self):
        # median filter
        self.denoised = median_filter(self.stack, footprint=np.ones((1, 1, 1)), mode="mirror")

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
        return self.z_max, self.y_max, self.x_max

    @cached_property
    def _filtered_center_plane(self):
        return median_filter(self.stack[self.z_max, :, :], 1, mode="mirror")

    @cached_property
    def x_fit(self):
        x_maxes = np.max(self._filtered_center_plane, axis=0)
        return self._fit_intensity(x_maxes, self.x_pixels, 1, 1)

    @property
    def x_max(self):
        return round(self.x_fit.x[1])

    @cached_property
    def y_fit(self):
        y_maxes = np.max(self._filtered_center_plane, axis=1)
        return self._fit_intensity(y_maxes, self.y_pixels, 1, 1)

    @property
    def y_max(self):
        return round(self.y_fit.x[1])

    @cached_property
    def z_fit(self):
        plane_maxes = np.max(self.denoised, axis=(1, 2))
        return self._fit_intensity(plane_maxes, self.planes, 1, 1)

    @property
    def z_max(self):
        return round(self.z_fit.x[1])

    @staticmethod
    def _fit_intensity(intensity, pixels, units, terms=1):
        x = np.arange(pixels) * units
        y = intensity

        if terms == 1:
            return least_squares(least_squares_residual,
                                 x0=[1, 1, 1],
                                 args=(x, y, single_term_gaussian))


# noinspection PyShadowingNames
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

    current_plane = psf.z_max

    # current plane
    ax1.imshow(psf.stack[current_plane, :, :], cmap="coolwarm")
    ax1.set_title("Current Plane")
    ax1.set_xticks([])
    ax1.set_yticks([])

    # center plane
    ax2.imshow(psf.stack[psf.z_max, :, :], cmap="coolwarm")
    ax2.set_title("Center Plane")
    ax2.set_xticks([])
    ax2.set_yticks([])

    ax3.imshow(psf.stack[:, :, psf.x_max], cmap="coolwarm")
    ax3.set_title("XZ")
    ax3.set_xticks([])

    ax4.imshow(psf.stack[:, psf.y_max, :], cmap="coolwarm")
    ax4.set_title("YZ")
    ax4.set_xticks([])

    def update():
        cp = plane_slider.value()
        ax1.imshow(psf.stack[cp, :, :], cmap="coolwarm")

    plane_slider = QtWidgets.QSlider(QtCore.Qt.Vertical)
    plane_slider.setRange(0, psf.planes)
    plane_slider.setSingleStep(1)
    plane_slider.setValue(psf.z_max)
    plane_slider.valueChanged.connect(update)
    # plane_slider.
    vbox = QtWidgets.QVBoxLayout()
    vbox.addWidget(plane_slider)
    fig.canvas.setLayout(vbox)

    return fig


def single_term_gaussian(c0, c1, c2, x, y):
    numerator = (x - c1)
    denominator = c2
    exponent = (numerator / denominator)**2
    exponent *= -1
    return c0 * np.exp(exponent)


def least_squares_residual(theta, x, y, func):
    c0, c1, c2 = theta
    pred = func(c0, c1, c2, x, y)
    return y - pred


psf = PSF(np.load("C:\\Users\\Darik\\psf.npy"), scaling=(0.5, 0.1, 0.1))

# fig = interactive_psf(psf)
