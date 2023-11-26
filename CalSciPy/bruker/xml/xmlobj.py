from __future__ import annotations
from typing import Any, Tuple
from abc import abstractmethod
from collections import ChainMap

from math import inf

from ...color_scheme import TERM_SCHEME
from ..._validators import validate_fields
from .. import CONSTANTS

# backport if necessary
from sys import version_info
if version_info.minor < 10:
    from _backports import dataclass, field
else:
    from dataclasses import dataclass, field

# BACKPORT INSPECTIONS IF NECESSARY
try:
    from inspect import get_annotations
except ImportError:
    from get_annotations import get_annotations


@dataclass(kw_only=True)
class _BrukerObject:
    """
    Abstract dataclass for bruker xml objects that provides methods for type-checking,
    type hints, and verbose printing.

    """
    def __post_init__(self: _BrukerObject) -> _BrukerObject:
        """
        Post-initialization method which conducts type-checking and other validation features

        """
        validate_fields(self)

    def __str__(self: _BrukerObject) -> str:
        """
        Modified dunder method such that the printing is more verbose and easier for human consumption

        Prints the dataclass name and each of its parameters, their values, and associated types

        """
        string_to_print = TERM_SCHEME(f"\n{self.__name__()}\n", "header")

        annotations_ = self.collect_annotations()

        for key, type_ in annotations_.items():
            if "MappingProxyType" in type_ or "dict" in type_:  # Not robust to nesting
                string_to_print += TERM_SCHEME(f"{key}", "emphasis")
                string_to_print += ":"
                string_to_print += TERM_SCHEME(f" ({type_})", "type")
                for nested_key in self.__dict__.get(key):
                    string_to_print += f"\n\t{nested_key}: {self.__dict__.get(key).get(nested_key)}"
                string_to_print += "\n"
            else:
                string_to_print += TERM_SCHEME(f"{key}", "emphasis")
                string_to_print += ":"
                string_to_print += f" {self.__dict__.get(key)}"
                string_to_print += TERM_SCHEME(f" ({type_})\n", "type")

        return string_to_print

    @staticmethod
    @abstractmethod
    def __name__() -> str:
        """
        Abstract modified static dunder method which returns the name of the dataclass

        """
        ...

    @classmethod
    def collect_annotations(cls: _BrukerObject) -> dict:
        """
        Class method that collects annotations of all parameters within the class and any of its parents

        """
        return dict(ChainMap(*(get_annotations(cls_) for cls_ in cls.__mro__)))

    def hint_types(self) -> None:
        """
        Verbose printing of type information to assist users in setting decodanda parameters
        """
        type_hints = TERM_SCHEME(f"\n{self.__name__()}\n", "header")

        annotations_ = self._collect_annotations()

        for key, type_ in annotations_.items():
            type_hints += TERM_SCHEME(f"{key}:", "emphasis")
            type_hints += TERM_SCHEME(f"{type_}\n", "type")

        print(type_hints)

    def __setattr__(self, key: str, value: Any) -> _BrukerObject:
        """
        Modified dunder method which performs type-checking on any change to attributes

        :param key: attribute key
        :type key: str
        :param value: associated value
        :type value: Any
        :rtype: _BrukerObject
        """
        self.__dict__[key] = value
        self.__post_init__()


@dataclass(kw_only=True)
class VoltageOutput(_BrukerObject):
    """
    Dataclass for a voltage recording

    """
    @staticmethod
    def __name__() -> str:
        return "VoltageOutput"


@dataclass(kw_only=True)
class VoltageRecording(_BrukerObject):
    """
    Dataclass for a voltage recording

    """
    @staticmethod
    def __name__() -> str:
        return "VoltageRecording"


@dataclass(kw_only=True)
class MarkPointSeriesElements(_BrukerObject):
    """
    Dataclass for a sequence of photostimulations
    """
    #: Tuple[MarkPointElement]: series of mark point elements
    marks: tuple = (None, )
    #: int: number of times this series is iterated
    iterations: int = field(default=1, metadata={"range": (1, inf)})
    #: float: delay between each series iteration (ms)
    iteration_delay: float = field(default=0.0, metadata={"range": (0.0, inf)})
    #: bool: whether to calculate functional map
    calc_funct_map: bool = False

    def __post_init__():
        if self.marks is not None:
            for idx, point in enumerate(self.points):
                if not isinstance(point, MarkPointElement):
                    raise TypeError(f"Marked point {idx} is not a MarkPointElement object")
        super().__post_init__()

    @staticmethod
    def __name__() -> str:
        return "MarkPointSeriesElements"


@dataclass(kw_only=True)
class MarkPointElement(_BrukerObject):
    """
    Dataclass for a specific marked point in a sequence of photostimulations
    """
    #: Tuple[GalvoPointElement]: Tuple of galvo point elements
    points: tuple = (None, )
    #: int: repetitions of this stimulation event
    repetitions: int = field(default=1, metadata={"range": (1, inf)})
    #: str: identity of uncaging laser
    uncaging_laser: str = "Uncaging"
    #: int: uncaging laser power
    uncaging_laser_power: float = field(default=0.0, metadata={"range": (0.0, 1000.0 / CONSTANTS.POWER_SCALE)})
    #: str: trigger frequency
    trigger_frequency: str = "None"
    # str: trigger selection id
    trigger_selection: str = "PFI0"
    #: int: number of triggers
    trigger_count: int = field(default=0, metadata={"range": (0, inf)})
    #: str: sync
    async_sync_frequency: str = "FirstRepetition"
    #: str: name of voltage output experiment
    voltage_output_category_name: str = "None"
    #: str: name of voltage recording experiment
    voltage_rec_category_name: str = "None"
    #: str: id of parameter set
    parameter_set: str = "CurrentSettings"

    def __post_init__():
        if self.points is not None:
            for idx, point in enumerate(self.points):
                if not isinstance(point, GalvoPointElement):
                    raise TypeError(f"Galvo point {idx} is not a GalvoPointElement object")
        super().__post_init__()

    @staticmethod
    def __name__() -> str:
        return "MarkPointElement"


@dataclass(kw_only=True)
class GalvoPointElement(_BrukerObject):
    """
    Dataclass for a specific galvo-stimulation for a specific marked point in a sequence of photostimulations
    """
    #: int: initial delay for stimulation
    initial_delay: float = field(default=1000.0, metadata={"range": (0, inf)})
    #: float: inter point delay
    inter_point_delay: float = field(default=0.12, metadata={"range": (0.12, inf)})
    #: int: duration of stimulation in ms
    duration: float = field(default=100.0, metadata={"range": {100.0, inf}})
    #: int: number of spiral revolutions
    spiral_revolutions: float = field(default=0.01, metadata={"range": (0.01, inf)})
    #: bool: whether to do all points at once
    all_points_at_once: bool = False
    #: str: id from galvo point list
    points: str = "Point 0"
    #: Tuple[int]: index from galvo point list
    indices: Tuple[int] = (0, )

    @staticmethod
    def __name__() -> str:
        return "GalvoPointElement"


@dataclass(kw_only=True)
class Point(_BrukerObject):
    """
     Dataclass for a specific point during galvo-stimulation for a specific marked point
     in a sequence of photostimulations

     """
    #: str: 1-order index in galvo point list
    index: int = field(default=0, metadata={"range": (0, inf)})
    #: float: normalized x position
    x: float = field(default=0.0, metadata={"range": (0.0, 1.0)})
    #: float: normalized y position
    y: float = field(default=0.0, metadata={"range": (0.0, 1.0)})
    #: bool: boolean indicating whether point is spiral
    is_spiral: bool = True
    #: float: width of spiral
    spiral_width: float = field(default=0.0, metadata={"range": (0.0, inf)})
    #: float: height of spiral
    spiral_height: float = field(default=0.0, metadata={"range": (0.0, inf)})
    #: float: size of spiral in microns
    spiral_size_in_microns: float = field(default=0.0, metadata={"range": (0.0, inf)})

    @staticmethod
    def __name__() -> str:
        return "Point"


@dataclass(kw_only=True)
class GalvoPointList(_BrukerObject):
    """
    Dataclass for a list of :class:`GalvoPoint`'s & :class:`GalvoPointGroup`'s

    """
    #: Tuple[Union[GalvoPoint, GalvoPointGroup]]
    galvo_points: tuple = None

    def __post_init__(self):
        """
        check that the galvo_points are correct classes

        """
        if self.galvo_points is not None:
            for idx, point in enumerate(self.galvo_points):
                if not isinstance(point, (GalvoPoint, GalvoPointGroup)):
                    raise TypeError(f"Galvo Point {idx} is not a GalvoPoint object")
        super().__post_init__()

    def __str__(self):
        return f" Galvo Point List containing {len(self.galvo_points)} marked points"

    @staticmethod
    def __name__() -> str:
        return "GalvoPointList"


@dataclass(kw_only=True)
class GalvoPointGroup(_BrukerObject):
    """
    Dataclass for a group of points during galvo-stimulation

    """
    #: Tuple[int, ...]: a tuple indexing the :class:`GalvoPoint`'s to be stimulated as part of this group
    indices: Tuple[int] = (0, )
    #: str: name of the group
    name: str = "Group 0"
    #: int: index of the photostimulation group (number of points + 1)
    index: int = 0
    #: str: type of activity
    activity_type: str = field(default="MarkPoints")
    #: str: stimulation laser ID
    uncaging_laser: str = "Uncaging"
    #: int: stimulation laser power (a.u.) scaled to the constant "power scale"
    uncaging_laser_power: float = field(default=0.0, metadata={"range": (0, 1000 / CONSTANTS.POWER_SCALE)})
    #: float: stimulation duration
    duration: float = field(default=100.0, metadata={"range": (0.0, inf)})
    #: bool: whether stimulation pattern is a spiral scan
    is_spiral: bool = True
    #: float: diameter of the spiral scaled to the number of x-pixels if spiral scan (radius?)
    #: TODO: DIAMETER OR RADIUS ???
    spiral_size: float = field(default=0.0, metadata={"range": (0.0, 2048.0)})
    #: float: number of spiral revolutions if spiral scan
    spiral_revolutions: float = field(default=0.01, metadata={"range": (0.0, inf)})
    #: float: relative z-position of the motor + ETL offset (um)
    z: float = field(default=0.0, metadata={"range": (-21500.0, 21500.0)})

    @staticmethod
    def __name__() -> str:
        return "GalvoPointGroup"


@dataclass(kw_only=True)
class GalvoPoint(_BrukerObject):
    """
     Dataclass for a specific point during galvo-stimulation for a specific marked point
     in a sequence of photostimulations

    """
    #: float: x position of the ROI scaled to voltage of the X Galvo
    x: float = field(default=0.0, metadata={"range": CONSTANTS.X_GALVO_VOLTAGE_RANGE})
    #: float: y position of the ROI scaled to the number of Y Galvo
    y: float = field(default=0.0, metadata={"range": CONSTANTS.Y_GALVO_VOLTAGE_RANGE})
    #: str: name of the roi
    name: str = field(default="Point 0")
    #: int: index of the roi in the galvo point list
    index: int = field(default=0, metadata={"range": (0, inf)})
    #: str: type of activity
    activity_type: str = field(default="MarkPoints")
    #: str: stimulation laser ID
    uncaging_laser: str = "Uncaging"
    #: int: stimulation laser power (a.u.) scaled to the constant "power scale"
    uncaging_laser_power: float = field(default=0.0, metadata={"range": (0, 1000 / CONSTANTS.POWER_SCALE)})
    #: float: stimulation duration
    duration: float = field(default=100.0, metadata={"range": (0.0, inf)})
    #: bool: whether stimulation pattern is a spiral scan
    is_spiral: bool = True
    #: float: diameter of the spiral scaled to the number of x-pixels if spiral scan
    spiral_size: float = field(default=0.0, metadata={"range": (0.0, 2048.0)})
    #: float: number of spiral revolutions if spiral scan
    spiral_revolutions: float = field(default=0.01, metadata={"range": (0.0, inf)})
    #: float: relative z-position of the motor + ETL offset (um)
    z: float = field(default=0.0, metadata={"range": (-21500.0, 21500.0)})

    @staticmethod
    def __name__() -> str:
        return "GalvoPoint"
