from __future__ import annotations
from typing import Tuple
from typing import NamedTuple
from memoization import cached
import numpy as np

# noinspection PyPackageRequirements
from scipy.signal import convolve
from scipy.ndimage import median_filter, affine_transform
from scipy.optimize import minimize, least_squares, OptimizeResult

import matplotlib

matplotlib.use("Qt5Agg")
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.backends.qt_compat import QtCore, QtWidgets, QtGui
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.gridspec import GridSpec


class PSF:
    """
    Class for experimental determination of point spread function

    :param stack: Planes x y-pixels x x-pixels imaging stack of unresolved fluorescent bead

    :param filter_shape: Shape of filter used for denoising

    :param scaling: Scaling of pixels

    :param downscale: Factor used for downscaling XY dimensions when calculating center plane

    """

    def __init__(self,
                 stack: np.ndarray,
                 downscale: float = 0.25,
                 filter_shape: Tuple[int, ...] = (1, 1, 1),
                 scaling: Tuple[float, ...] = (1, 1, 1)
                 ):

        #: :class:`np.ndarray <numpy.ndarray>`\: Planes x y-pixels x x-pixels imaging stack of
        #: unresolved fluorescent bead
        self.stack = self._validate_stack(stack)

        #: :class:`np.ndarray <numpy.ndarray>`\: Denoised pixel intensities with shape planes x y-pixels x x-pixels
        self.denoised = None

        #: :class:`np.ndarray <numpy.ndarray>`\: Convolved pixel intensities with shape planes x y-pixels x x-pixels
        self.convolved = None

        #: :class:`np.ndarray <numpy.ndarray>`\: Double-denoised pixel intensities at the center plane
        self.center = None

        #: :class:`float:`\: Factor used for downscaling XY dimensions when calculating center plane
        self.downscale = downscale if 0.0 < downscale <= 1.0 else 1.0

        #: :class:`Tuple <typing.Tuple>`\[:class:`int`\, ...]: Shape of filter used for denoising
        self.filter_shape = self._validate_filter_shape(filter_shape)

        #: :class:`Tuple <typing.Tuple>`\[:class:`float`\, ...]: Scaling of pixels
        self.scaling = self._validate_scaling(scaling)

        #: :class:`int`\: Number of planes in imaging stack
        self.planes = stack.shape[0]

        #: :class:`int`\: Number of y-pixels per plane in imaging stack
        self.y_pixels = stack.shape[1]

        #: :class:`int`\: Number of x-pixels per plane in imaging stack
        self.x_pixels = stack.shape[2]

        #: :class:`Tuple <typing.Tuple>`\[:class:`float`\, :class:`float`\]: Range of original data
        self.range = (np.min(stack), np.max(stack))

        # conduct denoising
        self._denoise()

        # conduct downscaling with subsequent convolution
        self._downscale()

        # secondary filter on center plane
        self.center = self._double_denoised(self.denoised, self.z_max)

    @property
    def bead(self):
        return self.z_max, self.y_max, self.x_max

    @property
    def fwhm(self):
        x = self._extract_fwhm(self.x_fit, self.x_scale)
        xz = self._extract_fwhm(self.xz_fit, self.z_scale)
        y = self._extract_fwhm(self.y_fit, self.y_scale)
        yz = self._extract_fwhm(self.yz_fit, self.z_scale)
        z = self._extract_fwhm(self.z_fit, self.z_scale)
        return FWHM(x, xz, y, yz, z)

    @property
    def x_fit(self):
        x_maxes = np.max(self.center, axis=0)
        return self._fit_intensity(x_maxes, self.x_pixels, self.scaling[2], 1)

    @property
    def x_max(self):
        return round(self.x_fit.x[1] / self.scaling[2])

    @property
    def x_scale(self):
        length = self.x_pixels * self.scaling[2]
        return np.linspace(-length / 2, length / 2, self.x_pixels)

    @property
    def xz_fit(self):
        planes = self.denoised[:, self.y_max, :]
        planes = np.max(planes, axis=-1)
        baselines = np.sort(planes)[1:5]
        planes = planes - np.mean(baselines)
        return self._fit_intensity(planes, self.planes, self.scaling[0], 1)

    @property
    def y_fit(self):
        y_maxes = np.max(self.center, axis=1)
        return self._fit_intensity(y_maxes, self.y_pixels, self.scaling[1], 1)

    @property
    def y_max(self):
        return round(self.y_fit.x[1] / self.scaling[1])

    @property
    def y_scale(self):
        length = self.y_pixels * self.scaling[1]
        return np.linspace(-length / 2, length / 2, self.y_pixels)

    @property
    def yz_fit(self):
        planes = self.denoised[:, :, self.x_max]
        planes = np.max(planes, axis=-1)
        baselines = np.sort(planes)[1:5]
        planes = planes - np.mean(baselines)
        return self._fit_intensity(planes, self.planes, self.scaling[0], 1)

    @property
    def z_fit(self):
        plane_maxes = np.max(self.convolved, axis=(1, 2))
        return self._fit_intensity(plane_maxes, self.planes, self.scaling[0], 1)

    @property
    def z_max(self):
        return round(self.z_fit.x[1] / self.scaling[0])

    @property
    def z_scale(self):
        length = self.planes * self.scaling[0]
        return np.linspace(-length / 2, length / 2, self.planes)

    @staticmethod
    @cached(max_size=2, order_independent=True)
    def _double_denoised(stack: np.ndarray, z_max: int) -> np.ndarray:
        return median_filter(stack[z_max, :, :], 1, mode="mirror")

    @staticmethod
    @cached(max_size=6, order_independent=True)
    def _extract_fwhm(fit: OptimizeResult, scale: np.ndarray) -> float:
        # adj
        adj_scale = scale - np.min(scale)
        model_intensities = single_term_gaussian(*fit.x, adj_scale, None)
        half_max = np.max(model_intensities) / 2
        residuals = np.abs(model_intensities - half_max)
        half_width = scale[np.argmin(residuals)]
        return np.abs(half_width * 2)

    @staticmethod
    @cached(max_size=6, order_independent=True)
    def _fit_intensity(intensity, pixels, units, terms=1):
        x = np.arange(pixels) * units
        y = intensity

        if terms == 1:
            return least_squares(least_squares_residual,
                                 x0=[1, 1, 1],
                                 args=(x, y, single_term_gaussian))

    @staticmethod
    def _validate_filter_shape(filter_shape: Tuple[int, ...]) -> Tuple[int, int, int]:
        if len(filter_shape) == 1:
            filter_shape *= 3
        elif len(filter_shape) == 2:
            filter_shape = (filter_shape[0], filter_shape[1], filter_shape[1])
        elif len(filter_shape) == 3:
            assert (filter_shape[1] == filter_shape[2]), "Requested incompatible filter shape"
        else:
            raise ValueError("Requested incompatible filter shape")
        return filter_shape

    @staticmethod
    def _validate_scaling(scaling: Tuple[float, ...]) -> Tuple[float, float, float]:
        if len(scaling) == 1:
            scaling *= 3
        elif len(scaling) == 2:
            scaling = (scaling[0], scaling[1], scaling[1])
        elif len(scaling) == 3:
            assert (scaling[1] == scaling[2]), "X & Y dimensions use different spatial scales"
        else:
            raise ValueError("Incompatible scaling provided")
        return scaling

    @staticmethod
    def _validate_stack(stack: np.ndarray) -> np.ndarray:
        assert (stack.ndim == 3), "Stack must be a three-dimensional array (planes x y-pixels x x-pixels)"
        stack -= np.min(stack)
        return stack.astype(np.float64)

    def _denoise(self):
        # median filter
        self.denoised = median_filter(self.stack, footprint=np.ones(self.filter_shape), mode="mirror")

        # subtract baseline
        self.denoised -= np.median(self.denoised)

    def _downscale(self):
        # bicubic interpolate downward
        scale_factor = 1 / self.downscale

        trans_matrix = np.zeros((4, 4))
        trans_matrix[0, 0] = 1
        trans_matrix[1, 1] = scale_factor
        trans_matrix[2, 2] = scale_factor
        trans_matrix[3, 3] = 1

        output_shape = (self.planes,
                        int(self.y_pixels // scale_factor),
                        int(self.x_pixels // scale_factor))

        denoised = np.ones(output_shape, dtype=np.float64) * scale_factor

        self.convolved = affine_transform(self.denoised,
                                          trans_matrix,
                                          output=denoised,
                                          output_shape=output_shape)

        # Convolve
        for plane in range(output_shape[0]):
            self.convolved[plane, :, :] = convolve(self.convolved[plane, :, :],
                                                   np.ones((2, 2)),
                                                   mode="same",
                                                   method="direct")


class FWHM(NamedTuple):
    x: float = None
    xz: float = None
    y: float = None
    yz: float = None
    z: float = None


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
    exponent = (numerator / denominator) ** 2
    exponent *= -1
    return c0 * np.exp(exponent)


def least_squares_residual(theta, x, y, func):
    c0, c1, c2 = theta
    pred = func(c0, c1, c2, x, y)
    return y - pred


psf = PSF(np.load("C:\\Users\\Darik\\psf.npy"), scaling=(0.5, 0.1, 0.1))

# fig = interactive_psf(psf)
