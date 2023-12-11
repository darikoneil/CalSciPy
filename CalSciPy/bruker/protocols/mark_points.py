from __future__ import annotations
from typing import Tuple, Sequence, Union, Optional, Mapping, List
from pathlib import Path
from collections import ChainMap
from copy import deepcopy
from itertools import chain

import numpy as np


from ...optogenetics import Photostimulation, StimulationGroup
from ...roi_tools import ROI
# noinspection PyProtectedMember

from ..._calculations import min_max_scale
from ..._validators import validate_keys
from .. import CONSTANTS
from ..xml.xmlobj import (GalvoPoint, GalvoPointList, GalvoPointGroup, MarkPointSeriesElements,
                          GalvoPointElement, MarkPointElement)
from ._helpers import write_protocol


"""
Collection of functions for generating protocols importable into PrairieView
"""


def generate_galvo_point_list(photostimulation: Photostimulation,
                              targets_only: bool = False,
                              parameters: Optional[dict] = None,
                              file_path: Optional[Path] = None,
                              name: Optional[str] = None,
                              z_offset: Optional[Union[float, Sequence[float]]] = None
                              ) -> GalvoPointList:
    """
    Generates a galvo point list to import identified or targeted ROIs into PrairieView.
    If you are using multiple planes, you must pass a z_offset for the stimulation laser for each joint
    imaging-stimulation plane. If you are stimulating only one plane and are using a device to modify the depth of the
    imaging and stimulation paths independently such as an electrically-tunable lens, then you only must either supply
    z_offset or enter the offset as "z" in the parameter dictionary

    :param photostimulation: instance of photostimulation object
    :param targets_only: flag indicating to generate a galvo point list containing only selected targets.
        this reduces complexity and is particularly useful if you are selecting a subset from a very large number of
        neurons
    :param parameters: stimulation parameters to override the galvo point defaults
    :param file_path: path to write galvo point list to
    :param name: name of protocol
    :param z_offset: z_offset for each plane
    :return: a galvo point list instance
    """

    # format for easy construction
    indices, points, rois, groups = _format_photostim(photostimulation, targets_only)

    # generate galvo point for each roi
    galvo_points = [_generate_galvo_point(roi=roi,
                                          index=index,
                                          name=f"ROI {point}",
                                          parameters=parameters,
                                          z_offset=z_offset)
                    for index, point, roi in zip(indices, points, rois)]

    # generate galvo point for each group if exists
    if photostimulation.num_groups > 0:
        index = [len(indices) + idx for idx in range(photostimulation.num_groups)]
        galvo_groups = [_generate_galvo_group(index=index,
                                              group=group,
                                              parameters=parameters)
                        for index, group in zip(index,
                                                groups)]
        galvo_points = tuple(chain.from_iterable([point for point in [galvo_points, galvo_groups]]))  # noqa: C416

    # instance galvo point list
    galvo_point_list = GalvoPointList(galvo_points=galvo_points)

    # write to file if requested
    if file_path is not None:
        write_protocol(galvo_point_list, file_path, ".gpl", name)

    return galvo_point_list


def generate_marked_points_protocol(photostimulation: Photostimulation,
                                    targets_only: bool = False,
                                    parameters: Optional[dict] = None,
                                    file_path: Optional[Path] = None,
                                    name: Optional[str] = None,
                                    z_offset: Optional[Union[float, Sequence[float]]] = None
                                    ) -> Tuple[MarkPointSeriesElements, GalvoPointList]:

    # we require a galvo point list
    gpl = generate_galvo_point_list(photostimulation,
                                    targets_only,
                                    parameters=parameters,
                                    file_path=file_path,
                                    name=name,
                                    z_offset=z_offset)

    mpl = _generate_mark_point_series(photostimulation,
                                      targets_only,
                                      parameters,
                                      file_path,
                                      name)

    return mpl, gpl


def _convert_parameters_relative_to_galvo_voltage(parameters: dict,
                                                  fov: Tuple[int, int] = CONSTANTS.FIELD_OF_VIEW_PIXELS,
                                                  pixels_per_micron: float = CONSTANTS.PIXELS_PER_MICRON,
                                                  power_scale: Tuple[float] = CONSTANTS.POWER_SCALE,
                                                  spiral_scale: Tuple[float] = CONSTANTS.SPIRAL_SCALE,
                                                  x_range: Tuple[float, float] = CONSTANTS.X_GALVO_VOLTAGE_RANGE,
                                                  y_range: Tuple[float, float] = CONSTANTS.Y_GALVO_VOLTAGE_RANGE,
                                                  inplace: bool = True
                                                  ) -> dict:
    if not inplace:
        parameters = deepcopy(parameters)

    if "uncaging_laser_power" in parameters:
        parameters["uncaging_laser_power"] *= power_scale

    if "spiral_size" in parameters:
        parameters["spiral_size"] *= pixels_per_micron * spiral_scale

    if "x" in parameters:
        # subtract one to account for rois being zero-indexed
        parameters["x"] = min_max_scale(parameters.get("x"), (0, fov[0] - 1), x_range)

    if "y" in parameters:
        # subtract one to account for rois being zero-indexed
        parameters["y"] = min_max_scale(parameters.get("y"), (0, fov[0] - 1), y_range)

    return parameters

def _format_photostim(photostimulation: Photostimulation,
                      targets_only: bool = False
                      ) -> Tuple[List[int], List[int], List[ROI], List[StimulationGroup]]:
    """
    Formats the relevant data such that we have an index from 0 to the number of included rois,
    the actual point relative to the group of total rois, (included & not included), a sequence of included rois,
    and a sequence of photostimulation groups

    :param photostimulation: photostimulation object
    :param targets_only: whether to use targets only when making the index
    :returns: index of included rois, index of included rois to total rois, sequence of included rois,
        sequence of photostimulation groups
    """
    # if we only want target roi's included
    if targets_only:
        points = photostimulation.stimulated_neurons
        indices = list(range(photostimulation.num_targets))
        rois = [photostimulation.rois.get(roi) for roi in points]
        groups = photostimulation.remapped_sequence
    else:
        indices = np.arange(photostimulation.num_neurons).tolist()
        points = np.arange(photostimulation.num_neurons).tolist()
        rois = [photostimulation.rois.get(roi) for roi in photostimulation.rois.keys()]
        groups = photostimulation.sequence
    return indices, points, rois, groups


def _generate_galvo_group(index: int,
                          group: StimulationGroup,
                          parameters: Optional[Mapping] = None,
                          z_offset: Optional[Union[float, Sequence[float]]] = None,
                          ) -> GalvoPointGroup:

    indices = tuple(group.ordered_index)

    name = group.name
    if name is None:
        tag = f"{indices}"[1:-1]
        name = "Group" + tag

    group_properties = dict(zip(
        ["indices", "name", "index"],
        [indices, name, index]
    ))

    # make sure parameters is not mutated, accomplished by breaking reference using deepcopy
    parameters = deepcopy(parameters)

    # make sure we only pass expected parameters since type checking will flag the unexpected
    parameters = validate_keys(GalvoPoint, parameters)

    # merge and allow parameters to override roi properties
    parameters = ChainMap(parameters, group_properties)

    # offset "z" if specified
    _offset_z(parameters, z_offset)

    # scale galvo point list parameters
    _convert_parameters_relative_to_galvo_voltage(parameters)

    return GalvoPointGroup(**parameters)


def _generate_galvo_point(roi: ROI,
                          index: Optional[int] = None,
                          name: Optional[str] = None,
                          parameters: Optional[Mapping] = None,
                          z_offset: Optional[float] = None
                          ) -> GalvoPoint:
    """
    Generates a single galvo point

    :param roi: ROI to convert into galvo point
    :param index: zero-based index of the ROI within the galvo point list
    :param name: by default Prairieview labels ROIs using a one-based index of the ROI within the galvo point list,
        but you can name them whatever you like
    :param parameters: stimulation parameters to override the galvo point defaults
    :param pixels_per_micron: number of pixels per micron for calculating spiral size
    :param power_scale: scaling factor used for calculating laser power
    :param reference_shape: reference reference_shape used for scaling roi coordinates (x, y)
    :param spiral_scale: scaling factor used for calculating spiral size
    :param x_range: voltage amplitude range of the x-galvo
    :param y_range: voltage amplitude range of the y-galvo
    :param z_offset: offset to relative z position if using multiple planes ("z" in .gpl file) & in microns
    :return: galvo point
    """
    # Make name if idx is provided and name is not
    if name is None and index is not None:
        name = f"Point {index}"

    # Retrieve and scale coordinates
    y, x = roi.centroid

    # flip y because prairieview used a different origin
    y = roi.reference_shape[0] - y

    # Retrieve spiral size
    spiral_size = roi.approximation.radius

    # Collect and merge with passed parameters. Allows overlap of things like spiral_size
    roi_properties = dict(zip(
        ["y", "x", "name", "index", "spiral_size"],
        [y, x, name, index, spiral_size]
    ))

    # make sure parameters is not mutated, accomplished by breaking reference using deepcopy
    parameters = deepcopy(parameters)

    # make sure we only pass expected parameters since type checking will flag the unexpected
    parameters = validate_keys(GalvoPoint, parameters)

    # merge and allow parameters to override roi properties
    parameters = ChainMap(parameters, roi_properties)

    # offset "z" if specified
    _offset_z(parameters, z_offset)

    # scale galvo point list parameters
    _convert_parameters_relative_to_galvo_voltage(parameters)

    return GalvoPoint(**parameters)


def _generate_mark_point_series(photostimulation: Photostimulation,
                                targets_only: bool,
                                parameters: dict,
                                file_path: Path,
                                name: str
                                ) -> MarkPointSeriesElements:

    # format for easy construction
    _, _, _, groups = _format_photostim(photostimulation, targets_only)

    if len(groups) > 1:
        # make mark point / galvo point elements
        for group in groups:
            mark_point_elements = tuple([_generate_mark_point_element(group, parameters) for group in groups])

        # make mark point series
        # noinspection PyUnboundLocalVariable
        mark_point_series = MarkPointSeriesElements(marks=mark_point_elements,
                                                    iterations=photostimulation.stim_sequence.repetitions,
                                                    iteration_delay=photostimulation.stim_sequence.delay)

        # write to file if requested
        if file_path is not None:
            write_protocol(mark_point_series, file_path, ".xml", name)

        return mark_point_series

    else:
        raise ValueError("Stimulation protocols must include at least one stimulation group")


def _generate_galvo_point_element(group: StimulationGroup,
                                  parameters: Optional[Mapping] = None,
                                  ) -> GalvoPointElement:
    indices = tuple(group.ordered_index)

    name = group.name
    if name is None:
        tag = f"{indices}"[1:-1]
        name = "Group" + tag

    initial_delay = group.delay

    inter_point_delay = group.point_interval

    group_properties = dict(zip(
        ["indices", "points", "initial_delay", "inter_point_delay"],
        [indices, name, initial_delay, inter_point_delay]
    ))

    # make sure parameters is not mutated, accomplished by breaking reference using deepcopy
    parameters = deepcopy(parameters)

    # make sure we only pass expected parameters since type checking will flag the unexpected
    parameters = validate_keys(GalvoPointElement, parameters)

    # merge and allow parameters to override roi properties
    parameters = ChainMap(parameters, group_properties)

    return GalvoPointElement(**parameters)


def _generate_mark_point_element(group: StimulationGroup,
                                 parameters: Optional[Mapping] = None
                                 ) -> MarkPointElement:
    # get number of repetitions
    repetitions = group.repetitions

    galvo_point_element = _generate_galvo_point_element(group, parameters)

    mark_point_properties = dict(zip(
        ["repetitions", "points"],
        [repetitions, (galvo_point_element, )]
    ))

    # make sure parameters is not mutated, accomplished by breaking reference using deepcopy
    parameters = deepcopy(parameters)

    # make sure we only pass expected parameters since type checking will flag the unexpected
    parameters = validate_keys(MarkPointElement, parameters)

    # convert values
    _convert_parameters_relative_to_galvo_voltage(parameters)

    # merge and allow parameters to override roi properties
    parameters = ChainMap(parameters, mark_point_properties)

    return MarkPointElement(**parameters)


def _offset_z(parameters: dict, z_offset: float = None) -> dict:

    if z_offset is not None:
        if "z" in parameters:
            parameters["z"] += z_offset
        else:
            parameters["z"] = z_offset
