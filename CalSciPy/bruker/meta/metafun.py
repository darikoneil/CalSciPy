from __future__ import annotations
from typing import Union, Tuple
from pathlib import Path
from xml.etree import ElementTree

from ..._validators import convert_permitted_types_to_required, validate_extension
from ..factories import BrukerElementFactory
from ..constants import CONSTANTS
from .metaobj import GalvoPointListMeta, MarkPointSeriesMeta, SavedMarkPointSeriesMeta


DEFAULT_PRAIRIEVIEW_VERSION = CONSTANTS.DEFAULT_PRAIRIEVIEW_VERSION


@convert_permitted_types_to_required(permitted=(str, Path), required=Path, pos=0)
def load_galvo_point_list(path: Union[str, Path]) -> GalvoPointListMeta:
    """
    Loads the marked points from a saved galvo point list

    :param path:

    :return:

    .. versionadded:: 0.8.1

    .. warning:: Currently untested

    """
    if path.is_file():
        return _import_gpl(path)
    elif not path.is_file():
        file = list(path.glob("*.gpl"))
        assert (len(file) == 1), f"Too many files meet parsing requirements: {file}"
        return _import_gpl(file)
    else:
        raise FileNotFoundError


@convert_permitted_types_to_required(permitted=(str, Path), required=str, pos=0)
def load_saved_mark_points(file_path: Union[str, Path],
                           version: str = DEFAULT_PRAIRIEVIEW_VERSION
                           ) -> SavedMarkPointSeriesMeta:
    """

    :param file_path: path to xml file
    :param version: version of prairieview
    :return: photostimulation metadata

    .. versionadded:: 0.8.1

    .. warning:: Currently untested

    """
    # We generally expected the file to be structured such that
    # File/
    # ├── MarkPointSeriesElements
    # │   └──MarkPointElement
    # │      ├──GalvoPointElement
    # │      |      └──Point
    #        └──GalvoPointElement (Empty)
    # However, by using configurable mappings we do have some wiggle room

    tree = ElementTree.parse(file_path)
    root = tree.getroot()

    # if it's imaging we can grab the version directly
    if "version" in root.attrib:
        version = root.attrib.get("version")

    bruker_element_factory = BrukerElementFactory(version)

    return SavedMarkPointSeriesMeta(root, factory=bruker_element_factory).marked_point_series


@convert_permitted_types_to_required(permitted=(str, Path), required=str, pos=0)
def load_mark_points(file_path: Union[str, Path],
                     version: str = DEFAULT_PRAIRIEVIEW_VERSION
                     ) -> Tuple[object, object]:
    """
    :param file_path: path to xml file
    :param version: version of prairieview
    :return: photostimulation metadata

    .. versionadded:: 0.8.1

    .. warning:: Currently untested

    """
    # We generally expected the file to be structured such that
    # File/
    # ├── MarkPointSeriesElements
    # │   └──MarkPointElement
    # │      ├──GalvoPointElement
    # │      |      └──Point
    #        └──GalvoPointElement (Empty)
    # However, by using configurable mappings we do have some wiggle room

    tree = ElementTree.parse(file_path)
    root = tree.getroot()

    # if it's imaging we can grab the version directly
    if "version" in root.attrib:
        version = root.attrib.get("version")

    bruker_element_factory = BrukerElementFactory(version)

    meta = MarkPointSeriesMeta(root, factory=bruker_element_factory)

    return meta.marked_point_series, meta.nested_points


@convert_permitted_types_to_required(permitted=(str, Path), required=str, pos=0)
@validate_extension(required_extension=".gpl", pos=0)
def _import_gpl(path: Union[str, Path], version: str = DEFAULT_PRAIRIEVIEW_VERSION) -> GalvoPointListMeta:
    """
    Implementation function for loading gpl files. Abstracted from the upper-level functions for cleaner logic.

    :param path: filepath of .gpl to import

    :return:
    """

    # We generally expected the file to be structured such that

    # File/
    # ├── GalvoPointList
    # │   |──GalvoPoint
    # |   |
    # |   |──GalvoPointGroup

    # However, by using configurable mappings we do have some wiggle room

    tree = ElementTree.parse(path)
    root = tree.getroot()

    # if it's imaging we can grab the version directly
    if "version" in root.attrib:
        version = root.attrib.get("version")

    factory = BrukerElementFactory(version)

    return GalvoPointListMeta(root=root, factory=factory).galvo_point_list
