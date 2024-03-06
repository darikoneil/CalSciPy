from __future__ import annotations
from typing import Mapping, Iterable, Any, List, Union, Tuple, NamedTuple
from types import MappingProxyType
from collections import namedtuple
from importlib import import_module
from xml.etree.ElementTree import Element
from numbers import Number

from .._validators import parse_tuple
from . import CONSTANTS
from .xml import load_mapping
# noinspection PyProtectedMember
from .xml.xmlobj import _BrukerObject


DEFAULT_PRAIRIEVIEW_VERSION = CONSTANTS.DEFAULT_PRAIRIEVIEW_VERSION
BRUKER_XML_OBJECT_MODULES = CONSTANTS.BRUKER_XML_OBJECT_MODULES


"""
These are factories for importing elements from bruker .xml files and generating elements from python objects
"""


class BrukerElementFactory:
    """
    Factory class for constructing bruker element objects

    :ivar element_class_mapping: dictionary mapping bruker xml objects to CalSciPy's bruker xml classes

    .. versionadded:: 0.8.1

    .. warning:: Currently untested
    """
    def __init__(self, version: str = DEFAULT_PRAIRIEVIEW_VERSION) -> BrukerElementFactory:
        """
        Factory class for constructing bruker element objects

        :param version: version of prairieview
        """
        self._collect_element_class_mapping(version)

    @staticmethod
    def _convert_key(old_key_: str) -> str:
        """
        Converts a key from CamelCase to python_standard


        :param old_key_: original CamelCase key
        """
        new_key = [old_key_[0].lower()]
        for _char in old_key_[1:]:
            if _char.isupper():
                new_key.append("_")
                new_key.append(_char.lower())
            else:
                new_key.append(_char)
        return "".join(new_key)

    @staticmethod
    def _type_conversion(attr: dict, type_annotations: dict) -> dict:
        """
        Duck-typing conversion of xml strings to expected types using the dataclass annotations

        :param attr: attributes containing keys and xml strings
        :param type_annotations: dictionary mapping attribute key and expected type
        """

        for key, value in attr.items():

            # Because bool("False") == True, here I specifically check for the "False" value
            if value == "False":
                attr[key] = False
            # Because we need to parse nested tuples more explicitly
            elif "Tuple[" in type_annotations.get(key):
                nested_type = type_annotations.get(key)
                nested_type = nested_type.split("[")[1].split("]")[0]
                nested_type = eval(nested_type)
                value = parse_tuple(value, nested_type)
                attr[key] = value
            else:
                attr[key] = eval("".join([type_annotations.get(key).lower(), "(value)"]))

    @classmethod
    def _map_attributes(cls: BrukerElementFactory, attr: dict) -> dict:
        """
        Generates new dictionary containing mapping of new keys with attribute values

        :param attr: attributes
        """
        mapping = cls._pythonize_keys(attr.keys())
        return {mapping.get(key): value for key, value in attr.items() if key in mapping}

    @classmethod
    def _pythonize_keys(cls: BrukerElementFactory, old_keys: Iterable[str]) -> MappingProxyType:
        """
        Converts keys from CamelCase to python_standard and generates a read-only dictionary
        mapping the original and new keys


        :param old_keys: original CamelCase keys
        """
        return {key: cls._convert_key(key) for key in old_keys}

    def constructor(self, element: Element) -> object:
        """
        Constructor method of bruker xml objects from xml Element

        :param element: xml Element
        """

        attr = self._map_attributes(element.attrib)
        bruker_object = self._identify_element_class(element)
        type_annotations = bruker_object.collect_annotations()
        self._type_conversion(attr, type_annotations)
        # noinspection PyArgumentList,PyCallingNonCallable
        return bruker_object(**attr)

    def _collect_element_class_mapping(self, version: str) -> BrukerElementFactory:
        """
        Loads appropriate mapping of xml tags and python objects

        :param version: prairieview version
        """
        self.element_class_mapping = load_mapping(version)

    def _identify_element_class(self, element: Element) -> object:
        """
        Identifies which bruker xml object to call given some element

        :param element: element in xml tree
        """
        try:
            assert (element.tag in self.element_class_mapping)
            return eval(self.element_class_mapping.get(element.tag))
        except NameError:
            return getattr(import_module(name=BRUKER_XML_OBJECT_MODULES, package="CalSciPy.bruker.xml"),
                           self.element_class_mapping.get(element.tag))
        except AssertionError:
            # here we raise a key error because he is missing, otherwise we could end up passing none to the eval)
            raise KeyError("Bruker object not present in element mapping")


class BrukerImageFactory:
    """
    Factory class for constructing bruker images

    .. versionadded:: 0.8.1

    .. warning:: Currently untested

    """
    @classmethod
    def create(cls: BrukerImageFactory, ch_pl_comb: Tuple[Tuple[int, int], ...]) -> NamedTuple:

        # make field names
        combinations = tuple([f"channel{ch}plane{pl}" for ch, pl in ch_pl_comb])

        return namedtuple("BrukerImages",
                          field_names=combinations,
                          defaults=[None, ] * len(combinations))


class BrukerXMLFactory:
    """
    Factory class for constructing CalSciPy's bruker xml objects from CalSciPy's protocol objects.

    :ivar element_class_mapping: dictionary mapping bruker xml objects to CalSciPy's bruker xml classes

    .. versionadded:: 0.8.1

    .. warning:: Currently untested

    """
    def __init__(self, version: str = DEFAULT_PRAIRIEVIEW_VERSION) -> BrukerXMLFactory:
        """
        Factory class for constructing bruker xml objects

        :param version: version of prairieview
        """
        self._collect_element_class_mapping(version)

    @staticmethod
    def _convert_key(old_key_: str) -> str:
        """
        Converts a key from python_standard to CamelCase

        :param old_key_: original python_standard key
        :return: key in CamelCase
        """
        key_parts = old_key_.split("_")
        key_parts = [new_key.capitalize() for new_key in key_parts]
        return "".join(key_parts)

    @staticmethod
    def _flatten_encoding(xml_encoding: List[Union[str, List[str]]]) -> List[str]:

        while len({type(line) for line in xml_encoding}) > 1:
            lines = []
            for line in xml_encoding:
                if isinstance(line, str):
                    lines.append(line)
                else:
                    lines.extend(line)
            xml_encoding = lines

        return xml_encoding

    @staticmethod
    def _generate_header() -> str:
        return '<?xml version="1.0" encoding="utf-8"?>'

    @staticmethod
    def _nested_validate_element(element: Any) -> bool:
        if isinstance(element, Iterable):
            failures = 0
            for child in element:
                try:
                    assert isinstance(child, _BrukerObject)
                except AssertionError:
                    failures += 1
            if failures == 0:
                return True

    @staticmethod
    def _pad_numeric_string(value: str) -> str:
        """
        Pads numeric string to be a minimum of 16-characters

        :param value: value to be padded
        :return: padded 16-character string
        """
        while len(value) <= 15:
            if "." in value:
                value += "0"
            else:
                value += "."
        # Make sure to include literal " characters
        return f'"{value}"'

    @staticmethod
    def _reverse_mapping(mapping: Mapping) -> MappingProxyType:
        """
        static method generates a read-only mapping where the values retrieve the keys

        :param mapping: the original mapping
        :return: a read-only mapping where the values retrieve the keys
        """
        return MappingProxyType({value: key for key, value in mapping.items()})

    @classmethod
    def _convert_float(cls: BrukerXMLFactory, value: float) -> str:
        """
        Converts a float value into a 16-character string representation

        :param value: float
        :return: 16-character string
        """
        # This is not efficient... but I don't think it really needs to be
        string_representation = f"{value}"

        if len(string_representation) < 16:
            return cls._pad_numeric_string(string_representation)
        elif len(string_representation) == 16:
            # make sure to include literal " characters
            return f'"{value}"'
        else:
            # make 15 character base tag
            tag = string_representation[:15]

            # Find sixteenth character by rounding the seventeenth
            sixteenth = int(string_representation[15])
            seventeenth = int(string_representation[16])

            # make sure to include literal " characters
            if seventeenth >= 5:
                tag += f'{sixteenth + 1}'
            else:
                tag += f'{string_representation[15]}'
            return f'"{tag}"'

    @classmethod
    def _parameter_to_xml(cls: BrukerXMLFactory, key: str, value: Any) -> str:
        """
        Convert a parameter to xml formatting

        :param key: key of parameter
        :param value: value of parameter
        :return: xml-formatted string
        """
        tag = f" {key}="
        if isinstance(value, Number):
            if type(value) == bool:
                tag += f'"{bool(value)}"'
            elif isinstance(value, float):
                tag += cls._convert_float(value)
            else:
                tag += f'"{value}"'
        elif isinstance(value, tuple):
            # special case, remove spaces and parenthesis in the tuple, check if only comma & drop it
            if len(value) == 1:
                tag_ = f"{value}"[1:-2]
            else:
                tag_ = f"{value}"[1:-1]
            tag_ = tag_.replace(" ", "")
            tag_ = '"' + tag_ + '"'
            tag += tag_
        else:
            tag += f'"{value}"'

        return tag

    @classmethod
    def validate_element(cls: BrukerXMLFactory, element: Any) -> Union[None, Exception]:
        try:
            assert isinstance(element, _BrukerObject)
        except AssertionError:
            try:
                assert cls._nested_validate_element(element)
            except AssertionError:
                try:
                    return NotImplementedError(f"{element.__name__()} is not a supported bruker object")
                except AttributeError:
                    return NotImplementedError(f"{type(element)} element is not a supported bruker object")

    @classmethod
    def _camel_case_keys(cls: BrukerXMLFactory, old_keys: Iterable[str]) -> dict:
        """
        Converts keys from python_standard to CamelCase and generates a read-only dictionary
        mapping the original and new keys

        :param old_keys: original python_standard keys
        :return: new keys in CamelCase
        """
        return {key: cls._convert_key(key) for key in old_keys}

    @classmethod
    def _has_children(cls: BrukerXMLFactory, attr: Mapping) -> bool:
        children = []
        for key, value in attr.items():
            if isinstance(value, _BrukerObject):
                children.append(key)
            if isinstance(value, Iterable):
                if cls._nested_validate_element(value):
                    children.append(key)
        if children:
            return children

    @classmethod
    def _map_attributes(cls: BrukerXMLFactory, attr: dict) -> dict:
        """
        Generates new dictionary containing mapping of new keys with attribute values

        :param attr: attributes
        """
        mapping = cls._camel_case_keys(attr.keys())
        return {mapping.get(key): value for key, value in attr.items() if key in mapping}

    def constructor(self, element: _BrukerObject) -> List:

        # first validate
        if self.validate_element(element):
            raise self.validate_element(element)

        # generate header
        lines = [self._generate_header(), ]

        # set iterator for indentation level
        level = 0

        # calculate and return
        constructed_elements = self.construct_element(element, level)

        for one_element in constructed_elements:
            lines.append(one_element)

        return self._flatten_encoding(lines)

    def construct_element(self, element: _BrukerObject, level: int = 0) -> str:

        tag = [self._generate_start_tag(element, level), ]

        # get camel case keys for writing to xml
        attr = self._map_attributes(vars(element))

        parameters = self._generate_parameters(attr)

        if parameters:
            tag[0] = "".join([tag[0], *parameters])

        if self._has_children(attr):
            tag[0] += ">"
            tag.append(self._generate_children(attr, level))
            tag.append(self._generate_end_tag(element, level))
        else:
            tag[0] += " />"

        return tag

    def _collect_element_class_mapping(self, version: str) -> BrukerXMLFactory:
        """
        Loads appropriate mapping of xml tags and python objects

        :param version: prairieview version
        """
        # we need to reverse the mapping such that the calscipy objects are the keys and the xml's the targets
        self.element_class_mapping = self._reverse_mapping(load_mapping(version))

    def _generate_children(self, attr: Mapping, level: int = 0) -> str:
        new_level = level + 2  # bump inward one indentation

        tags = []

        child_keys = self._has_children(attr)

        for key in child_keys:
            for child in attr.get(key):
                tags.extend(self.construct_element(child, new_level))

        return tags

    def _generate_end_tag(self, element: _BrukerObject, level: int = 0) -> str:
        """
        Generates the start tag of xml element

        :param element: element to be encoded as xml
        :param level: indentation level
        """
        tag = "\n"

        if level > 0:
            tag += " " * level * 2  # proper xml is 2 space indent of children

        tag += "</"
        # tag += element.xml_tag()
        tag += self.element_class_mapping.get(element.__name__())
        tag += ">"

        return tag

    def _generate_parameters(self, attr: Mapping) -> str:

        tags = []

        for key, value in attr.items():
            if self.validate_element(value):
                tags.append(self._parameter_to_xml(key, value))

        return "".join(tags)

    def _generate_start_tag(self, element: _BrukerObject, level: int = 0) -> str:
        """
        Generates the start tag of xml element

        :param element: element to be encoded as xml
        :param level: indentation level
        """
        tag = "\n"

        if level > 0:
            tag += " " * level  # proper xml is 2 space indent of children

        tag += "<"
        # tag += element.xml_tag()
        tag += self.element_class_mapping.get(element.__name__())
        return tag
