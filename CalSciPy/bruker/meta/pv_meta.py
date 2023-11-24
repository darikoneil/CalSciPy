from __future__ import annotations
from typing import Union
from pathlib import Path
from xml.etree import ElementTree

from _validators import convert_permitted_types_to_required, validate_extension


@convert_permitted_types_to_required(permitted=(str, Path), required=Path, pos=0)
def load_galvo_point_list(path: Union[str, Path]):
    """
    Loads the marked points from a saved galvo point list

    :param path:

    :return:
    """
    if path.is_file():
        return _import_gpl(path)
    elif not path.is_file():
        file = [file for file in path.glob("*.gpl")]
        assert (len(file) == 1), f"Too many files meet parsing requirements: {file}"
        return _import_gpl(file)
    else:
        raise FileNotFoundError


@convert_permitted_types_to_required(permitted=(str, Path), required=str, pos=0)
@validate_extension(required_extension=".gpl", pos=0)
def _import_gpl(path: Union[str, Path]):
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
