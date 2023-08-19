from __future__ import annotations
from pathlib import Path
import numpy as np

from CalSciPy.color_scheme import COLORS
from CalSciPy.optogenetics import Photostimulation

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.cm import get_cmap
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402
import seaborn as sns  # noqa: E402


def view_roi_overlay(photostimulation: Photostimulation):

    with plt.style.context("CalSciPy.main"):

        reference_image = photostimulation.reference_image
        reference_shape = photostimulation.reference_image.shape
        vi, vm = _reference_image_cutoffs(reference_image)

        fig, ax = _initialize_figure(reference_shape)

        ax.grid(visible=False)

        ax.imshow(reference_image, cmap="Spectral_r", vmin=vi, vmax=vm, interpolation="gaussian")

        for roi in photostimulation.rois.values():
            _generate_roi(roi, ax, lw=1.5, fill=False, edgecolor=COLORS.black)


def view_target_overlay(photostimulation: Photostimulation, targets=None):

    with plt.style.context("CalSciPy.main"):

        reference_image = photostimulation.reference_image
        reference_shape = photostimulation.reference_image.shape
        vi, vm = _reference_image_cutoffs(reference_image)

        fig, ax = _initialize_figure(reference_shape)

        ax.imshow(reference_image, cmap="Spectral_r", vmin=vi, vmax=vm, interpolation="gaussian")

        ax.grid(ls="--")

        if targets is None:

            default_list = np.setdiff1d(np.arange(photostimulation.num_neurons),
                                        list(photostimulation.stimulated_neurons)
                                        ).tolist()

            colors = _generate_colormap_spectrum(len(photostimulation.sequence), colormap="magma", alpha=.75)

            for idx, group in enumerate(photostimulation.sequence):
                _plot_targets(photostimulation,
                              group.ordered_index,
                              ax,
                              lw=1.5,
                              fill=False,
                              edgecolor=colors[idx]
                              )

            _plot_unstimulated(photostimulation,
                               default_list,
                               ax,
                               lw=1.5,
                               fill=False,
                               )

        else:
            default_list = np.setdiff1d(np.arange(photostimulation.num_neurons), list(targets)).tolist()

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

        colors = _generate_colormap_spectrum(photostimulation.num_neurons, colormap=colormap, alpha=0.75)

        for idx, roi in enumerate(photostimulation.rois.values()):
            _generate_roi(roi, ax, lw=1, edgecolor=COLORS.black, facecolor=colors[idx])


def view_targets(photostimulation: Photostimulation, targets=None):

    with plt.style.context("CalSciPy.main"):

        reference_shape = photostimulation.reference_image.shape

        reference_shape = photostimulation.reference_image.shape

        background_image = np.ones((*reference_shape, 3)) * COLORS.background

        fig, ax = _initialize_figure(reference_shape)

        ax.imshow(background_image)

        ax.grid(color=COLORS.black, ls="--")

        if targets is None:

            default_list = np.setdiff1d(np.arange(photostimulation.num_neurons),
                                        list(photostimulation.stimulated_neurons)
                                        ).tolist()

            colors = _generate_colormap_spectrum(len(photostimulation.sequence), alpha=0.75)

            for idx,group in enumerate(photostimulation.sequence):
                _plot_targets(photostimulation,
                              group.ordered_index,
                              ax,
                              lw=1,
                              edgecolor=COLORS.black,
                              facecolor=colors[idx]
                              )

            _plot_unstimulated(photostimulation,
                               default_list,
                               ax,
                               lw=1,
                               edgecolor=COLORS.black,
                               facecolor=COLORS.white
                               )

        else:
            default_list = np.setdiff1d(np.arange(photostimulation.num_neurons),
                                        list(targets)
                                        ).tolist()

            for idx, roi in enumerate(photostimulation.rois.values()):
                if idx in default_list:
                    _generate_roi(roi, ax, lw=1, edgecolor=COLORS.black, facecolor=COLORS.white)
                else:
                    _generate_roi(roi, ax, lw=1, edgecolor=COLORS.black, facecolor=COLORS.blue)


def _view_spiral_masked_targets(photostimulation: Photostimulation, targets=None):

    with plt.style.context("CalSciPy.main"):

        reference_shape = photostimulation.reference_image.shape

        reference_shape = photostimulation.reference_image.shape

        background_image = np.ones((*reference_shape, 3)) * COLORS.background

        fig, ax = _initialize_figure(reference_shape)

        ax.imshow(background_image)

        ax.grid(color=COLORS.black, ls="--")

        if targets is None:

            default_list = np.setdiff1d(np.arange(photostimulation.num_neurons),
                                        list(photostimulation.stimulated_neurons)
                                        ).tolist()

            colors = _generate_colormap_spectrum(len(photostimulation.sequence), alpha=0.75)

            for idx, group in enumerate(photostimulation.sequence):
                _plot_targets(photostimulation,
                              group.ordered_index,
                              ax,
                              lw=1,
                              edgecolor=COLORS.black,
                              facecolor=colors[idx]
                              )

            _plot_unstimulated(photostimulation,
                               default_list,
                               ax,
                               lw=1,
                               edgecolor=COLORS.black,
                               facecolor=COLORS.white
                               )

            for group in photostimulation.sequence:
                _plot_bound_mask(photostimulation,
                                 group.ordered_index,
                                 ax,
                                 lw=0,
                                 fill=True,
                                 facecolor=(*COLORS.black, 0.5)
                                 )


def view_spiral_targets(photostimulation: Photostimulation,
                        targets=None):\

    with plt.style.context("CalSciPy.main"):

        reference_shape = photostimulation.reference_image.shape

        reference_shape = photostimulation.reference_image.shape

        background_image = np.ones((*reference_shape, 3)) * COLORS.background

        fig, ax = _initialize_figure(reference_shape)

        ax.imshow(background_image)

        ax.grid(color=COLORS.black, ls="--")

        if targets is None:

            default_list = np.setdiff1d(np.arange(photostimulation.num_neurons),
                                        list(photostimulation.stimulated_neurons)
                                        ).tolist()

            colors = _generate_colormap_spectrum(len(photostimulation.sequence), alpha=0.75)

            for idx, group in enumerate(photostimulation.sequence):
                _plot_targets(photostimulation,
                              group.ordered_index,
                              ax,
                              lw=1,
                              edgecolor=COLORS.black,
                              facecolor=colors[idx]
                              )

            _plot_unstimulated(photostimulation,
                               default_list,
                               ax,
                               lw=1,
                               edgecolor=COLORS.black,
                               facecolor=COLORS.white
                               )

            for group in photostimulation.sequence:
                _plot_spiral(photostimulation,
                             group.ordered_index,
                             ax,
                             color=(*COLORS.black, 0.5),
                             lw=1
                             )


def _plot_spiral(photostimulation,
                 target_index,
                 axes,
                 **kwargs):
    for target in target_index:
        roi = photostimulation.rois.get(target)
        radius = roi.mask.bound_radius
        x, y = _make_spiral(*roi.coordinates[::-1], radius=radius)
        axes.plot(x, y, **kwargs)


def _make_spiral(x0, y0, theta=1000, radius=5.0):
    r = np.linspace(0, radius, 360)
    t = np.linspace(0, theta, 360)
    x = r * np.cos(np.radians(t))
    y = r * np.sin(np.radians(t))
    x += x0
    y += y0
    return x, y


def _plot_bound_mask(photostimulation,
                     target_index,
                     axes,
                     **kwargs):
    for target in target_index:
        roi = photostimulation.rois.get(target)
        pg = Polygon(roi.mask.bound_xy_vert, **kwargs)
        axes.add_patch(pg)


def _plot_targets(photostimulation, target_index, axes, **kwargs):
    for target in target_index:
        _generate_roi(photostimulation.rois.get(target),
                      axes,
                      **kwargs
                      )


def _plot_unstimulated(photostimulation,
                       default_list,
                       axes,
                       **kwargs):
    for neuron in default_list:
        _generate_roi(photostimulation.rois.get(neuron),
                      axes,
                      **kwargs
                      )


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
