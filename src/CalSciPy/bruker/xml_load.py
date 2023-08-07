from __future__ import annotations
from xml.etree import ElementTree
from .bruker_meta_objects import BrukerElementFactory
from .mark_points import PhotostimulationMeta
from .configuration_values import DEFAULT_PRAIRIEVIEW_VERSION


def read_mark_points_xml(file_path: str, version: str = DEFAULT_PRAIRIEVIEW_VERSION) -> PhotostimulationMeta:
    """

    :param file_path: path to xml file
    :type file_path: str or pathlib.Path
    :param version: version of prairieview
    :type version: str
    :return: photostimulation metadata
    :rtype: PhotostimulationMeta
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
