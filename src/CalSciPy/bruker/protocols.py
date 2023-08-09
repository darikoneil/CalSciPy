from __future__ import annotations
from typing import Tuple
from pathlib import Path
import numpy as np

from . import CONSTANTS
from .xml_objects import GalvoPoint, GalvoPointList
from .factories import BrukerXMLFactory
from ..optogenetics import Photostimulation
from ..roi_tools import ROI


def generate_galvo_point_list(photostimulation: Photostimulation,
                              parameters: Optional[dict] = None,
                              file_path: Path = None
                              ) -> GalvoPointList:
    """
    Generates a galvo point list to import identified or targeted ROIs into PrairieView

    :param photostimulation: instance of photostimulation object
    :param parameters: stimulation parameters to override the galvo point defaults
    :param file_path: path to write galvo point list to
    :return: a galvo point list instance
    """
    galvo_points = tuple([self.generate_galvo_point(idx, parameters) for idx in self.rois])

    galvo_point_list = GalvoPointList(galvo_points=galvo_points)

    if file_path is not None:
        write_protocol(galvo_point_list, file_path)

    return galvo_point_list


def write_protocol(pv, file_path: Path) -> None:
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
    :param reference_shape: reference shape used for scaling roi coordinates
    :param spiral_scale: scaling factor used for calculating spiral size
    :param x_range: voltage amplitude range of the x-galvo
    :param y_range: voltage amplitude range of the y-galvo
    :return: galvo point
    """
    # Make name if idx is provided and name is not
    if name is None and index is not None:
        name = f"Point {index}"

    # Scale laser power if in override parameters
    if "uncaging_laser_power" in parameters:
        parameters["uncaging_laser_power"] = _scale_power(parameters.get("uncaging_laser_power"))

    # Retrieve and scale coordinates
    y, x = roi.coordinates
    x_max, y_max = reference_shape
    y = _scale_coordinates(y, (0, y_max), y_range)
    x = _scale_coordinates(x, (0, x_max), x_range)

    # Retrieve and scale spiral size
    spiral_size = _scale_spiral(roi.mask.bound_radius)

    roi_properties = dict(zip(
        ["y", "x", "name", "index", "spiral_size"],
        [y, x, name, index, spiral_size]
    ))

    if parameters is not None:
        roi_properties = ChainMap(parameters, roi_properties)

    return GalvoPoint(**roi_properties)
