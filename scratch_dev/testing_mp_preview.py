from __future__ import annotations
from typing import Tuple, Union, Sequence
from numbers import Number
import os
from xml.etree import ElementTree

from CalSciPy.bruker.xml_load import read_mark_points_xml
from CalSciPy.bruker.bruker_meta_objects import BrukerElementFactory
from CalSciPy.bruker.mark_points import PhotostimulationMeta
from CalSciPy.bruker.configuration_values import DEFAULT_PRAIRIEVIEW_VERSION
import numpy as np

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import Polygon
from CalSciPy._interactive_visuals import COLORS
import seaborn as sns

from pathlib import Path
from scipy.spatial import ConvexHull
from scipy.ndimage import gaussian_filter


def setup_axes(ax):
    ax.set_xlabel("X Pixels")
    ax.set_ylabel("Y Pixels")
    ax.set_xlim(0, 488)
    ax.set_ylim(500, 0)
    ax.xaxis.set_major_locator(MultipleLocator(64))
    ax.yaxis.set_major_locator(MultipleLocator(64))
    ax.set_axisbelow(True)


def plot_on_reference(ax, stats, stimulation_list, reference_image):
    ax.set_title("Photostimulation Targets Group #1")
    lnr = np.ravel(reference_image)
    vi = np.percentile(lnr, 1)
    vm = 3 * np.percentile(lnr, 99)

    ax.imshow(reference_image, cmap="Spectral_r", vmax=vm, vmin=vi, interpolation="gaussian")

    for neuron in stimulation_list[0]:
        ax.add_patch(generate_polygon(stat[neuron], edgecolor=COLORS.red, lw=3, fill=False))


def plot_all_groups(ax, stats, stimulation_list, default_list):
    ax.set_title("Photostimulation Targets (ROI, Group)")
    ax.grid(color=COLORS.black, ls="--")
    blank = np.ones((500, 488, 3))
    blank *= COLORS.background
    ax.imshow(blank, interpolation="gaussian")

    for idx, neurons in enumerate(stimulation_list):
        for neuron in neurons:
            ax.add_patch(
                generate_polygon(stat[neuron], edgecolor=COLORS.black, lw=1.5, fill=True, facecolor=COLORS(idx)))

    for neuron in default_list:
        ax.add_patch(generate_polygon(stat[neuron], edgecolor=COLORS.black, lw=1.0, fill=True, facecolor=COLORS.white))


def preview_stimulation(stats, stimulation_list, default_list, reference_image):
    plt.style.use("CalSciPy.main")
    fig, axs = plt.subplots(1, 2)
    ax = axs[0]
    setup_axes(ax)
    plot_on_reference(ax, stats, stimulation_list, reference_image)
    ax = axs[1]
    setup_axes(ax)
    plot_all_groups(ax, stats, stimulation_list, default_list)


def multiple_group_without_replacement(sample_population, group_size, num_groups):
    selections = []
    for group in range(num_groups):
        selections.append(np.random.choice(sample_population, size=group_size, replace=False))
        sample_population = np.setdiff1d(sample_population, np.hstack(selections))
    return selections


def generate_polygon(stats, **kwargs):
    xpix = stats.get("xpix")[~stats.get("overlap")]
    ypix = stats.get("ypix")[~stats.get("overlap")]
    pts = np.vstack([ypix, xpix]).T
    hull = ConvexHull(pts)
    y, x = pts[hull.vertices, 0], pts[hull.vertices, 1]
    vtx_pts = np.vstack([x, y]).T
    return Polygon(vtx_pts, **kwargs)


ops_file = "D:\\DEM2\\preexposure\\results\\suite2p\\plane0\\ops.npy"
iscell_file = "D:\\DEM2\\preexposure\\results\\suite2p\\plane0\\iscell.npy"
stat_file = "D:\\DEM2\\preexposure\\results\\suite2p\\plane0\\stat.npy"

ops = np.load(ops_file, allow_pickle=True).item()
iscell = np.load(iscell_file, allow_pickle=True)
stat = np.load(stat_file, allow_pickle=True)

neuronal_index = np.where(iscell[:, 0] == 1)[0]

num_neurons = 5
num_groups = 5

stimulation_list = multiple_group_without_replacement(neuronal_index, num_neurons, num_groups)

default_list = np.setdiff1d(neuronal_index, np.hstack(stimulation_list))

reference_image = ops.get("Vcorr")


preview_stimulation(stat, stimulation_list, default_list, reference_image)
