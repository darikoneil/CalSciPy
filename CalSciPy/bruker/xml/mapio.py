from __future__ import annotations
import os
from typing import Mapping
from json import dump, load
from types import MappingProxyType
from pathlib import Path


MAP_DIR = Path(os.path.realpath(__file__)).parents[0].joinpath("xml_mappings")


def write_mapping(mapping: Mapping, version: str) -> None:
    """
    Write mapping of prairieview xml objects to their
    respective pyprairieview objects to .json

    :param mapping: Mapping the xml tag and python object

    :type mapping: :class:`Mapping <typing.Mapping>`

    :param version: Version of prairieview

    .. versionadded:: 0.8.1

    .. warning:: Currently untested

    """
    # replace dots in version path with underscores
    version = version.replace(".", "_")
    filename = Path(MAP_DIR).joinpath(version).with_suffix(".json")
    with open(filename, "w") as file:
        dump(mapping, file, indent=4)


def load_mapping(version: str) -> MappingProxyType:
    """
    Loads mapping of prairieview xml objects to their respective pyprairieview objects from .json

    :param version: Version of prairieview

    :returns: Read-only mapping the xml tag and python object

    :rtype: :class:`MappingProxyType <types.MappingProxyType.>`

    .. versionadded:: 0.8.1

    .. warning:: Currently untested

    """
    # replace dots in version path with underscores
    version = version.replace(".", "_")
    filename = Path(MAP_DIR).joinpath(version).with_suffix(".json")
    with open(filename, "r") as file:
        mapping = MappingProxyType(load(file))
    return mapping
