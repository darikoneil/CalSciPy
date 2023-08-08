from __future__ import annotations
from typing import Any, Tuple
from dataclasses import dataclass, field
from abc import abstractmethod
from collections import ChainMap
from inspect import get_annotations
from PPVD.style import TerminalStyle
from prettytable import PrettyTable, ALL
from .validation import validate_fields
from math import inf


# These require python 3.10?
@dataclass(kw_only=True)
class _BrukerObject:
    """
    Abstract dataclass for bruker xml objects that provides methods for type-checking,
    type hints, and verbose printing.

    """
    def __post_init__(self: _BrukerObject) -> _BrukerObject:
        """
        Post-initialization method which conducts type-checking and other validation features

        :rtype: _BrukerObject
        """
        validate_fields(self)

    def __str__(self: _BrukerObject) -> str:
        """
        Modified dunder method such that the printing is more verbose and easier for human consumption

        Prints the dataclass name and each of its parameters, their values, and associated types
        :rtype: str
        """
        string_to_print = f"\n{TerminalStyle.YELLOW}{TerminalStyle.BOLD}{TerminalStyle.UNDERLINE}" \
                          f"{self.__name__()}{TerminalStyle.RESET}\n"

        annotations_ = self.collect_annotations()

        for key, type_ in annotations_.items():
            if "MappingProxyType" in type_ or "dict" in type_:  # Not robust to nesting
                string_to_print += f"{TerminalStyle.YELLOW}{TerminalStyle.BOLD}{key}{TerminalStyle.RESET}:"
                string_to_print += f"{TerminalStyle.BLUE} ({type_}){TerminalStyle.RESET}"
                for nested_key in self.__dict__.get(key):
                    string_to_print += f"\n\t{nested_key}: {self.__dict__.get(key).get(nested_key)}"
                string_to_print += "\n"
            else:
                string_to_print += f"{TerminalStyle.YELLOW}{TerminalStyle.BOLD}{key}: " \
                                   f"{TerminalStyle.RESET}{self.__dict__.get(key)}" \
                                   f"{TerminalStyle.BLUE} ({type_}){TerminalStyle.RESET}\n"
        return string_to_print

    @staticmethod
    @abstractmethod
    def __name__() -> str:
        """
        Abstract modified static dunder method which returns the name of the dataclass

        :rtype: str
        """
        return "MarkPointObjectFactory"

    @classmethod
    def collect_annotations(cls: _BrukerObject) -> dict:
        """
        Class method that collects annotations of all parameters within the class and any of its parents

        """
        return dict(ChainMap(*(get_annotations(cls_) for cls_ in cls.__mro__)))

    def hint_types(self: _BrukerObject) -> None:
        """
        Prints type hints for each parameter

        :rtype: None
        """
        pretty_table = PrettyTable(hrules=ALL, vrules=ALL, float_format=".2f")
        title = f"{TerminalStyle.BOLD}{TerminalStyle.YELLOW}{self.__name__()}{TerminalStyle.RESET}"
        type_string = f"{TerminalStyle.BOLD}{TerminalStyle.YELLOW}Type{TerminalStyle.RESET}"
        pretty_table.field_names = [title, type_string]
        pretty_table.align[title] = "l"
        pretty_table.align[type_string] = "c"
        annotations_ = self.collect_annotations()
        for key, type_ in annotations_.items():
            pretty_table.add_row([f"{TerminalStyle.BOLD}{key}{TerminalStyle.RESET}",
                                  f" {type_} "])
        print(pretty_table.get_string())

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
class MarkPointSeriesElements(_BrukerObject):
    """
    Dataclass for a sequence of photostimulations
    """
    #: Tuple[object]: series of mark point elements
    marks: Tuple[object]
    #: int: number of times this series is iterated
    iterations: int = field(default=1, metadata={"range": (1, inf)})
    #: float: delay between each series iteration (ms)
    iteration_delay: float = field(default=0.0, metadata={"range": (0.0, inf)})
    #: bool: whether to calculate functional map
    calc_funct_map: bool = False

    @staticmethod
    def __name__() -> str:
        return "MarkPointSeriesElements"


@dataclass(kw_only=True)
class MarkPointElement(_BrukerObject):
    """
    Dataclass for a specific marked point in a sequence of photostimulations
    """
    #: object: Tuple of galvo point elements
    points: Tuple[object]
    #: int: repetitions of this stimulation event
    repetitions: int = field(default=1, metadata={"range": (1, inf)})
    #: str: identity of uncaging laser
    uncaging_laser: str = "Uncaging"
    #: int: uncaging laser power
    uncaging_laser_power: int = field(default=0, metadata={"range": (0, 1000)})
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
    voltage_rec_category_name: str = "Current"
    #: str: id of parameter set
    parameter_set: str = "CurrentSettings"

    @staticmethod
    def __name__() -> str:
        return "MarkPointElement"


@dataclass(kw_only=True)
class GalvoPointElement(_BrukerObject):
    """
    Dataclass for a specific galvo-stimulation for a specific marked point in a sequence of photostimulations
    """
    #: int: initial delay for stimulation
    initial_delay: int = field(default=0, metadata={"range": (0, inf)})
    #: float: inter point delay
    inter_point_delay: float = field(default=0.0, metadata={"range": (0.0, inf)})
    #: int: duration of stimulation in ms
    duration: float = field(default=0, metadata={"range": {0.0, inf}})
    #: int: number of spiral revolutions
    spiral_revolutions: float = field(default=0, metadata={"range": (0.0, inf)})
    #: bool: whether to do all points at once
    all_points_at_once: bool = False
    #: str: id from galvo point list
    points: str = "Point 0"
    #: int: index from galvo point list
    indices: int = field(default=0, metadata={"range": (0, inf)})

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
    Dataclass for a list of galvo points

    """
    galvo_points: Tuple[object]

    def __post_init__(self):
        for idx, point in enumerate(self.galvo_points):
            if not isinstance(point, GalvoPoint):
                raise TypeError(f"Galvo Point {idx} is not a GalvoPoint object")
        super().__post_init__()

    def __str__(self):
        return f" Galvo Point List containing {len(self.galvo_points)} ROIs"

    @staticmethod
    def __name__() -> str:
        return "GalvoPointList"


@dataclass(kw_only=True)
class GalvoPoint(_BrukerObject):
    """
     Dataclass for a specific point during galvo-stimulation for a specific marked point
     in a sequence of photostimulations

    :ivar z: relative z-position of the motor + ETL offset (um)
    :type z: float
    """
    x: float = field(default=0.0, metadata={"range": (0.0, 2048.0)})
    y: float = field(default=0.0, metadata={"range": (0.0, 2048.0)})
    name: str = field(default="Point 0")
    index: int = field(default=0, metadata={"range": (0, inf)})
    activity_type: str = field(default="MarkPoints")
    uncaging_laser: str = "Uncaging"
    uncaging_laser_power: int = field(default=0, metadata={"range": (0, 1000)})
    duration: float = field(default=100.0, metadata={"range": (0.0, inf)})
    is_spiral: bool = True
    spiral_size: float = field(default=0.0, metadata={"range": (0.0, 2048.0)})
    spiral_revolutions: float = field(default=0.0, metadata={"range": (0.0, inf)})
    z: float = field(default=0.0, metadata={"range": (-8192.0, 8192.0)})

    # @staticmethod
    # def xml_tag() -> str:
    #    return "PVGalvoPoint"

    @staticmethod
    def __name__() -> str:
        return "GalvoPoint"
