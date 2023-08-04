from __future__ import annotations
from typing import List
from abc import abstractmethod
from types import MappingProxyType
from .xml_mappings.xml_mapping import load_mapping
from xml.etree import ElementTree
from xml.etree.ElementTree import Element
from importlib import import_module
from .configuration_values import BRUKER_XML_OBJECT_MODULES, DEFAULT_PRAIRIEVIEW_VERSION


class BrukerMeta:
    def __init__(self, root: ElementTree, factory: object) -> BrukerMeta:
        """
        Abstract class for bruker metadata

        :param root: root of xml element tree
        :type root: xml.etree.ElementTree
        :param factory: factory for building metadata
        :type factory: object
        :rtype: BrukerMeta
        """
        self._build_meta(root, factory)
        self._extra_actions()

    @staticmethod
    @abstractmethod
    def __name__() -> str:
        """
        Abstract modified static dunder method which returns the name of the dataclass

        :rtype: str
        """
        return ""

    @abstractmethod
    def _build_meta(self, root: ElementTree, factory: object) -> BrukerMeta:
        """
        Abstract method for building metadata from the root of the xml's element tree using a factory class

        :param root: root of xml element tree
        :type root: xml.etree.ElementTree
        :param factory: factory for building metadata
        :type factory: object
        :rtype: BrukerMeta
        """
        pass

    @abstractmethod
    def _extra_actions(self, *args, **kwargs) -> BrukerMeta:
        """
        Abstract method for any additional actions here

        :rtype: BrukerMeta
        """
        pass

    @abstractmethod
    def generate_protocol(self, path: str) -> None:
        """
        Generates a protocol for the metadata to be imported into prairieview

        :param path: path to write protocol
        :type path: str or pathlib.Path
        :rtype: None
        """
        pass


class BrukerElementFactory:
    def __init__(self, version: str = DEFAULT_PRAIRIEVIEW_VERSION) -> BrukerElementFactory:
        """
        Factory class for constructing bruker element objects

        :param version: version of prairieview
        :type version: str
        :rtype: BrukerElementFactory
        """
        self._collect_element_class_mapping(version)

    @staticmethod
    def _convert_key(old_key_: str) -> str:
        """
        Converts a key from CamelCase to python_standard


        :param old_key_: original CamelCase key
        :type old_key_: str
        :rtype: str
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
        :type attr: dict
        :param type_annotations: dictionary mapping attribute key and expected type
        :type type_annotations: dict
        :rtype: dict
        """
        # Because bool("False") == True, here I specifically check for the "False" value
        return {key: (eval("".join([type_annotations.get(key), "(value)"])) if value != "False" else False)
                for key, value in attr.items()}

    @classmethod
    def _map_attributes(cls: BrukerElementFactory, attr: dict) -> dict:
        """
        Generates new dictionary containing mapping of new keys with attribute values

        :param attr: attributes
        :type attr: dict
        :rtype: dict
        """
        mapping = cls._pythonize_keys(attr.keys())
        return {mapping.get(key): value for key, value in attr.items() if key in mapping}

    @classmethod
    def _pythonize_keys(cls: BrukerElementFactory, old_keys: List[str]) -> MappingProxyType:
        """
        Converts keys from CamelCase to python_standard and generates a read-only dictionary
        mapping the original and new keys


        :param old_keys: original CamelCase keys
        :type old_keys: List[str]
        :rtype: MappingProxyType
        """
        return {key: cls._convert_key(key) for key in old_keys}

    def constructor(self, element: Element) -> object:
        """
        Constructor method of bruker xml objects from xml Element

        :param element: xml Element
        :type element: Element
        :rtype:object
        """

        attr = self._map_attributes(element.attrib)
        bruker_object = self._identify_element_class(element)
        type_annotations = bruker_object.collect_annotations()
        attr = self._type_conversion(attr, type_annotations)
        # noinspection PyArgumentList,PyCallingNonCallable
        return bruker_object(**attr)

    def _collect_element_class_mapping(self, version: str) -> BrukerElementFactory:
        """
        Loads appropriate mapping of xml tags and python objects

        :param version: prairieview version
        :type version: str
        :rtype: BrukerElementFactory
        """
        self.element_class_mapping = load_mapping(version)

    def _identify_element_class(self, element: Element) -> object:
        """
        Identifies which bruker xml object to call given some element

        :param element: element in xml tree
        :type element: Element
        :rtype: object
        """
        try:
            assert (element.tag in self.element_class_mapping)
            return eval(self.element_class_mapping.get(element.tag))
        except NameError:
            return getattr(import_module(name=BRUKER_XML_OBJECT_MODULES, package="CalSciPy.bruker"),
                           self.element_class_mapping.get(element.tag))
        except AssertionError:
            # here we raise a key error because he is missing, otherwise we could end up passing none to the eval)
            raise KeyError("Bruker object not present in element mapping")


class BrukerXMLFactory:
    def __init__(self, version=DEFAULT_PRAIRIEVIEW_VERSION) -> BrukerXMLFactory:
        """
        Factory class for constructing bruker xml objects

        :param version: version of prairieview
        :type version: str
        :rtype: BrukerXMLFactory
        """
        self._collect_element_class_mapping(version)

    def _collect_element_class_mapping(self, version: str) -> BrukerXMLFactory:
        """
        Loads appropriate mapping of xml tags and python objects

        :param version: prairieview version
        :type version: str
        :rtype: BrukerXMLFactory
        """
        self.element_class_mapping = load_mapping(version)

