from __future__ import annotations
from typing import Union
from pathlib import Path
from xml.etree import ElementTree

from bruker import CONSTANTS
from bruker.meta.meta_objects import PhotostimulationMeta
from bruker.factories import BrukerElementFactory
from _validators import convert_permitted_types_to_required


DEFAULT_PRAIRIEVIEW_VERSION = CONSTANTS.DEFAULT_PRAIRIEVIEW_VERSION


"""
A collection of functions for importing / converting PrairieView collected data
"""


@convert_permitted_types_to_required(permitted=(str, Path), required=str, pos=0)
def load_mark_points(file_path: Union[str, Path], version: str = DEFAULT_PRAIRIEVIEW_VERSION) -> PhotostimulationMeta:
    """

    :param file_path: path to xml file
    :param version: version of prairieview
    :return: photostimulation metadata
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

    return PhotostimulationMeta(root, factory=bruker_element_factory)
