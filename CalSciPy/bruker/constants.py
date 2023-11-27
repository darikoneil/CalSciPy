from __future__ import annotations
from configparser import ConfigParser
from pathlib import Path
from typing import Tuple
from numbers import Number

# backport if python < 3.10
from sys import version_info
if version_info.minor < 10:
    from .._backports import dataclass
else:
    from dataclasses import dataclass


"""Configuration constants that describe the user's PrairieView setup"""


_config_parser = ConfigParser()

_path = Path(__file__).with_name("config.ini")

_config = _config_parser.read(_path)

_OBJECTIVE_FLAG = _config_parser.get("OBJECTIVE", "OBJECTIVE")

_PIXELS_PER_MICRON = _config_parser.get(_OBJECTIVE_FLAG, "PIXELS_PER_MICRON")

_MAGNIFICATION = _config_parser.get(_OBJECTIVE_FLAG, "MAGNIFICATION")


def _calculate_field_of_view(value: Number,
                             pixels_per_micron: float = _PIXELS_PER_MICRON
                             ) -> float:
    """
    Calculate FOV using number of pixels per micron

    :param value: value to be scaled
    :param pixels_per_micron: pixels per micron
    :return: field of view length on one axis
    """
    if not isinstance(value, float):
        value = float(value)

    if not isinstance(pixels_per_micron, float):
        pixels_per_micron = float(pixels_per_micron)

    return value * pixels_per_micron


def _format_amplitude_range(value: Number) -> Tuple[float, float]:
    """
    Formats voltage amplitude of galvo in range

    :param value: galvo amplitude
    :return: voltage amplitude range
    """

    if not isinstance(value, float):
        value = float(value)

    return -1 * value, value


def _scale_to_magnification(value: Number,
                            magnification: float = _MAGNIFICATION
                            ) -> float:
    """
    Scales some value according to the magnification

    :param value: value to be scaled
    :param magnification: magnification of objective
    :return: scaled value
    """
    if not isinstance(value, float):
        value = float(value)

    if not isinstance(magnification, float):
        magnification = float(magnification)

    return value / magnification


@dataclass(frozen=True)
class CONSTANTS:
    """
    Immutable dataclass containing constants describing prairieview setup

    """
    #: flag for file containing xml objects (for forward compatibility, ignore)
    BRUKER_XML_OBJECT_MODULES: str = _config_parser.get("CALSCIPY", "BRUKER_XML_OBJECT_MODULES")

    #: version of prairieview software
    DEFAULT_PRAIRIEVIEW_VERSION: str = _config_parser.get("PRAIRIEVIEW", "VERSION")

    #: field of view in pixels (X, Y)
    FIELD_OF_VIEW_PIXELS: Tuple[int, int] \
        = (int(_config_parser.get("SOFTWARE", "FIELD_OF_VIEW_X_PIXELS")),
           int(_config_parser.get("SOFTWARE", "FIELD_OF_VIEW_Y_PIXELS"))
           )

    #: field of view in microns (X, Y)
    FIELD_OF_VIEW_MICRONS: Tuple[float, float] = \
        tuple([_scale_to_magnification(value) for value in
               [_calculate_field_of_view(_config_parser.get("SOFTWARE", "FIELD_OF_VIEW_X_PIXELS")),
                _calculate_field_of_view(_config_parser.get("SOFTWARE", "FIELD_OF_VIEW_Y_PIXELS"))]
               ])

    #: magnification used during imaging
    MAGNIFICATION: float = float(_config_parser.get(_OBJECTIVE_FLAG, "MAGNIFICATION"))

    #: objective used during imaging
    OBJECTIVE: str = _OBJECTIVE_FLAG

    #: pixels per micron
    PIXELS_PER_MICRON: float = _scale_to_magnification(_config_parser.get(_OBJECTIVE_FLAG, "PIXELS_PER_MICRON"))

    #: scaling used for laser power (1 unit)
    POWER_SCALE: float = 1 / float(_config_parser.get("HARDWARE", "POWER_SCALE"))

    #: scaling used for spiral size (1 unit)
    SPIRAL_SCALE: float = 1 / _scale_to_magnification(_config_parser.get(_OBJECTIVE_FLAG, "SPIRAL_SCALE"))

    #: voltage amplitude range of the X-Galvo
    X_GALVO_VOLTAGE_RANGE: Tuple[float, float] = \
        tuple([_scale_to_magnification(value)
               for value in _format_amplitude_range(_config_parser.get("HARDWARE", "X_GALVO_VOLTAGE"))])

    #: voltage amplitude range of the Y-Galvo
    Y_GALVO_VOLTAGE_RANGE: Tuple[float, float] = \
        tuple([_scale_to_magnification(value)
               for value in _format_amplitude_range(_config_parser.get("HARDWARE", "Y_GALVO_VOLTAGE"))])
