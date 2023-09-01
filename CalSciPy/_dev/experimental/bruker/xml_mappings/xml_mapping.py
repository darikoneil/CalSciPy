from __future__ import annotations

import os
from json import dump, load
from types import MappingProxyType
import pathlib


MAP_DIR = pathlib.Path(os.path.realpath(__file__)).parents[0]


def write_mapping(mapping: dict, version: str) -> None:
    """
    Write mapping of prairieview xml objects to their
    respective pyprairieview objects to .json

    :param mapping: dictionary mapping the xml tag and python object
    :type mapping: dict
    :param version: version of prairieview
    :type version: str
    :rtype: None
    """
    # replace dots in version path with underscores
    version = version.replace(".", "_")
    version = "".join([version, ".json"])
    filename = os.path.join(MAP_DIR, version)
    with open(filename, "w") as file:
        dump(mapping, file, indent=4)


def load_mapping(version: str) -> MappingProxyType:
    """
    Load mapping of prairieview xml objects to their
    respective pyprairieview objects from .json

    :param version: version of prairieview
    :type version: str
    :return: read-only mapping the xml tag and python object
    :rtype: MappingProxyType
    """
    version = version.replace(".", "_")
    version = "".join([version, ".json"])
    filename = os.path.join(MAP_DIR, version)
    with open(filename, "r") as file:
        mapping = MappingProxyType(load(file))
    return mapping
