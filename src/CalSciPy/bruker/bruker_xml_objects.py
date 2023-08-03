from __future__ import annotations
from typing import Any
from dataclasses import dataclass, field
from abc import abstractmethod
from collections import ChainMap
from inspect import get_annotations
from PPVD.style import TerminalStyle
from prettytable import PrettyTable, ALL
from .validation import validate_fields


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
        string_to_print = ""
        string_to_print += f"\n{self.__name__()}\n"

        annotations_ = self.collect_annotations()

        for key, type_ in annotations_.items():
            if "MappingProxyType" in type_ or "dict" in type_:  # Not robust to nesting
                string_to_print += f"{TerminalStyle.YELLOW}{TerminalStyle.BOLD}{key}{TerminalStyle.RESET}:"
                string_to_print += f"{TerminalStyle.YELLOW} ({type_}){TerminalStyle.RESET}"
                for nested_key in self.__dict__.get(key):
                    string_to_print += f"\n\t{nested_key}: {self.__dict__.get(key).get(nested_key)}"
                string_to_print += "\n"
            else:
                string_to_print += f"{TerminalStyle.YELLOW}{TerminalStyle.BOLD}{key}: " \
                                   f"{TerminalStyle.RESET}{self.__dict__.get(key)}" \
                                   f"{TerminalStyle.YELLOW} ({type_}){TerminalStyle.RESET}\n"
        return string_to_print

    @staticmethod
    @abstractmethod
    def xml_tag() -> str:
        """
        Abstract static method which returns the tag of the object within bruker xml

        """
        return ""

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

        :rtype: dict
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
    #: int: number of times this series is iterated
    iterations: int = 1
    #: float: delay between each series iteration (ms)
    iteration_delay: float = 0.12
    #: bool: whether to calculate functional map
    calc_funct_map: bool = False

    @staticmethod
    def xml_tag() -> str:
        return "PVMarkPointSeriesElements"

    @staticmethod
    def __name__() -> str:
        return "MarkPointSeriesElements"


@dataclass(kw_only=True)
class MarkPointElement(_BrukerObject):
    """
    Dataclass for a specific marked point in a sequence of photostimulations
    """
    #: int: repetitions of this stimulation event
    repetitions: int = 10
    #: str: identity of uncaging laser
    uncaging_laser: str = "Uncaging"
    #: int: uncaging laser power
    uncaging_laser_power: int = 1000
    #: str: trigger frequency
    trigger_frequency: str = "None"
    # str: trigger selection id
    trigger_selection: str = "PFI0"
    #: int: number of triggers
    trigger_count: int = 50
    #: str: sync
    async_sync_frequency: str = "FirstRepetition"
    #: str: name of voltage output experiment
    voltage_output_category_name: str = "None"
    #: str: name of voltage recording experiment
    voltage_rec_category_name: str = "Current"
    #: str: id of parameter set
    parameter_set: str = "CurrentSettings"

    @staticmethod
    def xml_tag() -> str:
        return "PVMarkPointElement"

    @staticmethod
    def __name__() -> str:
        return "MarkPointElement"


@dataclass(kw_only=True)
class GalvoPointElement(_BrukerObject):
    """
    Dataclass for a specific galvo-stimulation for a specific marked point in a sequence of photostimulations
    """
    #: int: initial delay for stimulation
    initial_delay: int = 0
    #: float: inter point delay
    inter_point_delay: float = 0.0
    #: float: duration of stimulation in ms
    duration: float = 0
    #: int: number of spiral revolutions
    spiral_revolutions: int = 0
    #: bool: whether to do all points at once
    all_points_at_once: bool = False
    #: str: id from galvo point list
    points: str = "Point 1"
    #: int: index from galvo point list
    indices: int = 1

    @staticmethod
    def xml_tag() -> str:
        return "PVGalvoPointElement"

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
    index: int = 1
    #: float: normalized x position
    x: float = 0.5
    #: float: normalized y position
    y: float = 0.5
    #: bool: boolean indicating whether point is spiral
    is_spiral: bool = True
    #: float: width of spiral
    spiral_width: float = 0.0340461221379181
    #: float: height of spiral
    spiral_height: float = 0.0340461221379181
    #: float: size of spiral in microns
    spiral_size_in_microns: float = 20

    @staticmethod
    def xml_tag() -> str:
        return "Point"

    @staticmethod
    def __name__() -> str:
        return "Point"


@dataclass(kw_only=True)
class GalvoPoint(_BrukerObject):
    """
     Dataclass for a specific point during galvo-stimulation for a specific marked point
     in a sequence of photostimulations

     """
    x: float = -0.314262691377921
    y: float = 9000.0
    name: str = field(default="Point 5")
    index: int = 4
    activity_type: str = field(default="MarkPoints")
    uncaging_laser: str = "Uncaging"
    uncaging_laser_power: int = field(default=0)
    duration: float = 100.0
    is_spiral: bool = True
    spiral_size: float = 0.338219758489175
    spiral_revolutions: float = 0.0
    z: float = 54.8692328473932

    @staticmethod
    def xml_tag() -> str:
        return "PVGalvoPoint"

    @staticmethod
    def __name__() -> str:
        return "GalvoPoint"
