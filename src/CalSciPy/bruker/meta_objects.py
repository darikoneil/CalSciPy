from __future__ import annotations
from abc import abstractmethod
from xml.etree import ElementTree


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
    def generate_protocol(self, path: str) -> None:
        """
        Generates a protocol for the metadata to be imported into prairieview

        :param path: path to write protocol
        :type path: str or pathlib.Path
        :rtype: None
        """
        pass

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
