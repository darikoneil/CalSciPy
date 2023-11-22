from __future__ import annotations
from typing import Tuple, Optional, Union, List, Any
from operator import eq
from pathlib import Path
from itertools import product
from xml.etree import ElementTree

import numpy as np
import pandas as pd
import cv2
from PPVD.parsing import convert_permitted_types_to_required, find_num_unique_files_given_static_substring, \
    find_num_unique_files_containing_tag
from PPVD.validation import validate_extension
from tqdm import tqdm as tq

from . import CONSTANTS
from .meta_objects import PhotostimulationMeta
from .factories import BrukerElementFactory
from .._calculations import generate_blocks
from .._files import calculate_frames_per_file
from ..io_tools import _load_single_tif, _save_single_tif
from .._backports import PatternMatching


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
