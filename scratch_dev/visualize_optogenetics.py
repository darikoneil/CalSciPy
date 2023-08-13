from __future__ import annotations
from pathlib import Path
import numpy as np

from CalSciPy._interactive_visuals import COLORS
from CalSciPy.optogenetics import Photostimulation

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.cm import get_cmap
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402
import seaborn as sns  # noqa: E402


def multiple_group_without_replacement(sample_population, group_size, num_groups):
    selections = []
    for group in range(num_groups):
        selections.append(np.random.choice(sample_population, size=group_size, replace=False))
        sample_population = np.setdiff1d(sample_population, np.hstack(selections))
    return selections


def view_roi_overlay(photostimulation: Photostimulation):

    with plt.style.context("CalSciPy.main"):

        reference_image = photostimulation.reference_image
        reference_shape = photostimulation.reference_image.shape
        vi, vm = _reference_image_cutoffs(reference_image)

        fig, ax = _initialize_figure(reference_shape)

        ax.grid(visible=False)

        ax.imshow(reference_image, cmap="Spectral_r", vmin=vi, vmax=vm, interpolation="gaussian")

        for roi in photostimulation.rois.values():
            _generate_roi(roi, ax, lw=1, fill=False, edgecolor=COLORS.black)


def view_target_overlay(photostimulation: Photostimulation, targets):

    with plt.style.context("CalSciPy.main"):

        reference_image = photostimulation.reference_image
        reference_shape = photostimulation.reference_image.shape
        vi, vm = _reference_image_cutoffs(reference_image)

        fig, ax = _initialize_figure(reference_shape)

        ax.imshow(reference_image, cmap="Spectral_r", vmin=vi, vmax=vm, interpolation="gaussian")

        default_list = np.setdiff1d(np.arange(photostimulation.total_neurons), list(targets)).tolist()

        for idx, roi in enumerate(photostimulation.rois.values()):
            if idx in default_list:
                _generate_roi(roi, ax, lw=1, fill=False, edgecolor=COLORS.black)
            else:
                _generate_roi(roi, ax, lw=1, fill=False, edgecolor=COLORS.red)


def view_rois(photostimulation: Photostimulation, colormap="Spectral_r"):

    with plt.style.context("CalSciPy.main"):

        reference_shape = photostimulation.reference_image.shape
        background_image = np.ones((*reference_shape, 3)) * COLORS.background
        fig, ax = _initialize_figure(reference_shape)

        ax.imshow(background_image)

        colors = _generate_colormap_spectrum(photostimulation.total_neurons, colormap=colormap, alpha=0.75)

        for idx, roi in enumerate(photostimulation.rois.values()):
            _generate_roi(roi, ax, lw=1, edgecolor=COLORS.black, facecolor=colors[idx])


def view_targets(photostimulation: Photostimulation, targets):

    with plt.style.context("CalSciPy.main"):

        reference_shape = photostimulation.reference_image.shape

        reference_shape = photostimulation.reference_image.shape

        # background_image = np.ones((*reference_shape, 3)) * COLORS.background

        fig, ax = _initialize_figure(reference_shape)

        # ax.imshow(background_image)

        ax.grid(color=COLORS.black, ls="--")

        # colors = _generate_colormap_spectrum(photostimulation.neurons, colormap=colormap, alpha=0.75)

        default_list = np.setdiff1d(np.arange(photostimulation.total_neurons), list(targets)).tolist()

        for idx, roi in enumerate(photostimulation.rois.values()):
            if idx in default_list:
                _generate_roi(roi, ax, lw=1, edgecolor=COLORS.black, facecolor=COLORS.white)
            else:
                _generate_roi(roi, ax, lw=1, edgecolor=COLORS.black, facecolor=COLORS.blue)


def _initialize_figure(reference_shape, title: str = "ROIs", x_label: str = "", y_label: str = ""):

    fig, ax = plt.subplots(1, 1)
    ax.set_xlim(0, reference_shape[1])
    ax.set_ylim(0, reference_shape[0])
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.xaxis.set_major_locator(MultipleLocator(64))
    ax.yaxis.set_major_locator(MultipleLocator(64))
    ax.set_axisbelow(True)

    return fig, ax


def _generate_colormap_spectrum(num_colors: int,
                                colormap="Spectral_r",
                                alpha: float = None
                                ) -> Tuple[Tuple[float, float, float, float]]:

    colormap = get_cmap(colormap)

    cmap_points = np.linspace(0, 1, num_colors).tolist()

    colors = tuple([colormap(point) for point in cmap_points])

    if alpha is not None:
        colors = tuple([(*color[:3], alpha) for color in colors])

    return colors


def _generate_roi(roi, axes, **kwargs):
    pg = Polygon(roi.xy_vert, **kwargs)
    axes.add_patch(pg)


def _reference_image_cutoffs(reference_image):
    flattened_reference = np.ravel(reference_image)
    vi = np.percentile(flattened_reference, 1)
    vm = np.percentile(flattened_reference, 99)
    vm += (3 * abs(vm))
    return vi, vm
