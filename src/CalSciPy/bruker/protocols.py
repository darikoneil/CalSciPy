from __future__ import annotations
from typing import Tuple, Any, Sequence, Union, Optional
from numbers import Number
from pathlib import Path
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
                              parameters: Optional[dict] = None,
                              file_path: Path = None,
                              name: Optional[str] = None,
                              z_offset: Union[float, Sequence[float]] = None
                              ) -> GalvoPointList:
    """
    Generates a galvo point list to import identified or targeted ROIs into PrairieView.
    If you are using multiple planes or a device to modify the depth of the imaging and stimulation paths independently,
    you must pass a z_offset for the stimulation laser for each joint imaging-stimulation plane

    :param photostimulation: instance of photostimulation object
    :param parameters: stimulation parameters to override the galvo point defaults
    :param file_path: path to write galvo point list to
    :param name: name of protocol
    :param z_offset: z_offset for each plane
    :return: a galvo point list instance
    """
    # Generate each galvo point
    galvo_points = tuple([self.generate_galvo_point(roi=roi, index=index, parameters=parameters, z_offset=z_offset)
                          for index, roi in enumerate(self.rois)])

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
                          index: Optional[int] = None,
                          name: Optional[str] = None,
                          parameters: Optional[dict] = None,
                          pixels_per_micron: float = CONSTANTS.PIXELS_PER_MICRON,
                          power_scale: Tuple[float] = CONSTANTS.POWER_SCALE,
                          reference_shape: Tuple[int, int] = CONSTANTS.FIELD_OF_VIEW_PIXELS,
                          spiral_scale: Tuple[float] = CONSTANTS.SPIRAL_SCALE,
                          x_range: Tuple[float, float] = CONSTANTS.X_GALVO_VOLTAGE_RANGE,
                          y_range: Tuple[float, float] = CONSTANTS.Y_GALVO_VOLTAGE_RANGE,
                          z_offset: float = None
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

    # Scale laser power if in override parameters
    if "uncaging_laser_power" in parameters:
        parameters["uncaging_laser_power"] = _scale_power(parameters.get("uncaging_laser_power"))

    # Scale spiral if in override parameters
    if "spiral_size" in parameters:
        parameters["spiral_size"] = _scale_spiral(parameters.get("spiral_size"))

    # Offset "z" if provided
    if "z" in parameters and z_offset is not None:
        if isinstance(z_offset, Number):
            parameters["z"] += z_offset
        if isinstance(z_offset, Sequence):
            this_plane = roi.plane
            parameters["z"] += z_offset[this_plane]

    # Retrieve and scale coordinates
    y, x = roi.coordinates
    x, y = _scale_coordinates(x, y, reference_shape)

    # Retrieve and scale spiral size
    spiral_size = _scale_spiral(roi.mask.bound_radius)

    roi_properties = dict(zip(
        ["y", "x", "name", "index", "spiral_size"],
        [y, x, name, index, spiral_size]
    ))

    if parameters is not None:
        roi_properties = ChainMap(parameters, roi_properties)

    return GalvoPoint(**roi_properties)


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
