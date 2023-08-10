from __future__ import annotations
from typing import Tuple, Any, Sequence, Union, Optional
from numbers import Number
from pathlib import Path
from collections import ChainMap
from copy import deepcopy

import numpy as np

from PPVD.validation import validate_filename
from PPVD.parsing import convert_permitted_types_to_required

from . import CONSTANTS
from .xml_objects import GalvoPoint, GalvoPointList, _BrukerObject
from .factories import BrukerXMLFactory
from ..optogenetics import Photostimulation
from ..roi_tools import ROI


"""
Collection of functions for generating protocols importable into PrairieView
"""


_DEFAULT_PATH = Path.cwd().joinpath("prairieview_protocol.xml")


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
    imaging-stimulation plane. If you are stimulating only one plane and using a device to modify the depth of the
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

    # Reduce number of galvo points required if desired
    if targets_only:
        permitted_points = photostimulation.targets
    else:
        permitted_points = np.arange(photostimulation.neurons).tolist()

    # Generate each galvo point
    galvo_points = tuple([_generate_galvo_point(roi=roi, index=index, parameters=parameters, z_offset=z_offset)
                          for index, roi in enumerate(photostimulation.rois.values()) if index in permitted_points])

    # Instance galvo point list
    galvo_point_list = GalvoPointList(galvo_points=galvo_points)

    if file_path is not None:
        write_protocol(galvo_point_list, file_path, ext=".gpl")

    return galvo_point_list


@validate_filename(pos=1)
@convert_permitted_types_to_required(permitted=(str, Path), required=Path, pos=1)
def write_protocol(protocol: _BrukerObject,
                   file_path: Path = _DEFAULT_PATH,
                   ext: str = ".xml",
                   name: Optional[str] = None) -> None:
    """
    Write prairieview protocol to file

    :param protocol: prairieview object
    :param file_path: file path for saving file
    :param name: name of protocol
    :param ext: file extension for saving file
    """

    # If name  provided, append name
    if name is not None:
        file_path = file_path.joinpath(name)

    # If no extension, append ext
    if file_path.suffix == '':
        file_path = file_path.with_suffix(ext)

    factory = BrukerXMLFactory()
    lines = factory.constructor(pv)
    with open(file_path, "w+") as file:
        for line in lines:
            file.write(line)


def _generate_galvo_point(roi: ROI,
                          pixels_per_micron: float = CONSTANTS.PIXELS_PER_MICRON,
                          power_scale: Tuple[float] = CONSTANTS.POWER_SCALE,
                          reference_shape: Tuple[int, int] = CONSTANTS.FIELD_OF_VIEW_PIXELS,
                          spiral_scale: Tuple[float] = CONSTANTS.SPIRAL_SCALE,
                          x_range: Tuple[float, float] = CONSTANTS.X_GALVO_VOLTAGE_RANGE,
                          y_range: Tuple[float, float] = CONSTANTS.Y_GALVO_VOLTAGE_RANGE,
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
    :param reference_shape: reference shape used for scaling roi coordinates (x, y)
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
    y, x = roi.coordinates

    # Retrieve spiral size
    spiral_size = roi.mask.bound_radius

    # Collect and merge with passed parameters. Allows overlap of things like spiral_size
    roi_properties = dict(zip(
        ["y", "x", "name", "index", "spiral_size"],
        [y, x, name, index, spiral_size]
    ))

    # make sure parameters is not mutated, accomplished by breaking reference using deepcopy
    parameters = deepcopy(parameters)

    # merge and allow parameters to override roi properties
    parameters = ChainMap(parameters, roi_properties)

    # Offset "z" if specified
    if "z" in parameters and z_offset is not None:
        if isinstance(z_offset, Number):
            parameters["z"] += z_offset
        if isinstance(z_offset, Sequence):
            this_plane = roi.plane
            parameters["z"] += z_offset[this_plane]

    _scale_galvo_point_parameters(parameters)

    return GalvoPoint(**parameters)


def _scale_galvo_point_parameters(parameters: dict,
                                  pixels_per_micron: float = CONSTANTS.PIXELS_PER_MICRON,
                                  power_scale: Tuple[float] = CONSTANTS.POWER_SCALE,
                                  reference_shape: Tuple[int, int] = CONSTANTS.FIELD_OF_VIEW_PIXELS,
                                  spiral_scale: Tuple[float] = CONSTANTS.SPIRAL_SCALE,
                                  x_range: Tuple[float, float] = CONSTANTS.X_GALVO_VOLTAGE_RANGE,
                                  y_range: Tuple[float, float] = CONSTANTS.Y_GALVO_VOLTAGE_RANGE,
                                  ) -> dict:

    # Scale coordinates if specified in parameters
    if "x" in parameters and "y" in parameters and reference_shape is not None:
        x = parameters.get("x")
        y = parameters.get("y")
        x, y = _scale_coordinates((x, y), reference_shape)
        parameters["x"] = x
        parameters["y"] = y

    # Scale laser power if specified in parameters
    if "uncaging_laser_power" in parameters:
        parameters["uncaging_laser_power"] = _scale_power(parameters.get("uncaging_laser_power"))

    # Scale spiral if specified in parameters
    if "spiral_size" in parameters:
        parameters["spiral_size"] = _scale_spiral(parameters.get("spiral_size"))


def _scale_coordinates(coordinates: Tuple[Number, Number],
                       fov: Tuple[int, int] = CONSTANTS.FIELD_OF_VIEW_PIXELS,
                       x_range: Tuple[float, float] = CONSTANTS.X_GALVO_VOLTAGE_RANGE,
                       y_range: Tuple[float, float] = CONSTANTS.Y_GALVO_VOLTAGE_RANGE,
                       ) -> Tuple[float, float]:
    """
    Scale x, y coordinates to scanner voltages

    :param coordinates: x, y coordinates
    :param fov: field of view in pixels
    :param x_range: voltage amplitude of x galvo
    :param y_range: voltage amplitude of y galvo
    :return: coordinates scaled to scanner voltages
    """

    # adjust for zero-index
    fov = tuple([pixels - 1 for pixels in fov])

    return tuple([_scale_coordinate(coordinate, (0, pixels), voltage_range)
                  for coordinate, pixels, voltage_range
                  in zip(coordinates, fov, (x_range, y_range))])


def _scale_coordinate(value: Number, old_range: Tuple[int, int], new_range: Tuple[float, float]) -> float:
    """
    Scale coordinate to new range

    :param value: coordinate
    :param old_range: old range
    :param new_range: new range
    :return: scaled value
    """
    old_min, old_max = old_range
    new_min, new_max = new_range
    return new_min + ((value - old_min) * (new_max - new_min)) / (old_max - old_min)


def _scale_power(value: Number, multiplier: float = CONSTANTS.POWER_SCALE) -> float:
    """
    Scales laser power to correct range in .gpl files

    :param value: unscaled laser power value
    :param multiplier: power scale multiplier constant
    :return: scaled value
    """
    return value * multiplier


def _scale_spiral(value: Number,
                  pixels_per_micron: float = CONSTANTS.PIXELS_PER_MICRON,
                  multiplier: float = CONSTANTS.SPIRAL_SCALE
                  ) -> float:
    """
    Scales spiral diameter to correct range in .gpl files

    :param value: unscaled spiral size value
    :param pixels_per_micron: number of pixels per micron constant
    :param multiplier: spiral size multiplier constant
    :return: scaled value
    """
    return value * pixels_per_micron * multiplier
