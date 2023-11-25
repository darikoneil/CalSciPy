from __future__ import annotations
from typing import Tuple
from abc import abstractmethod
from xml.etree import ElementTree
from collections import ChainMap

import numpy as np
from scipy.spatial import ConvexHull

from ...color_scheme import TERM_SCHEME
from ...roi_tools import calculate_mask
from ..xml.xml_objects import GalvoPointList, MarkPointElement, MarkPointSeriesElements  # womp womp, has to be exception atm


class _BrukerMeta:
    """
    Abstract class for bruker metadata

    """
    def __init__(self, root: ElementTree, factory: object) -> _BrukerMeta:
        """
        Abstract class for bruker metadata

        :param root: root of xml element tree
        :param factory: factory for building metadata
        """
        self._build_meta(root, factory)
        self._extra_actions()

    @staticmethod
    @abstractmethod
    def __name__() -> str:
        """
        Abstract modified static dunder method which returns the name of the dataclass

        """
        ...

    @abstractmethod
    def _build_meta(self, root: ElementTree, factory: object) -> _BrukerMeta:
        """
        Abstract method for building metadata from the root of the xml's element tree using a factory class

        :param root: root of xml element tree
        :param factory: factory for building metadata
        """
        ...

    @abstractmethod
    def _extra_actions(self, *args, **kwargs) -> _BrukerMeta:
        """
        Abstract method for any additional actions here

        """
        ...


class GalvoPointListMeta(_BrukerMeta):
    def __init__(self, root: ElementTree, factory: object):
        """
        Metadata object for a galvo point list saved from mark points in pyprairieview

        """
        self.galvo_point_list = None
        super().__init__(root, factory)

    def _build_meta(self, root: ElementTree, factory: object) -> _BrukerMeta:
        self.galvo_point_list = (
            GalvoPointList(galvo_points=[factory.constructor(child) for idx, child in enumerate(root)]))

    def _extra_actions(self, *args, **kwargs) -> _BrukerMeta:
        return 0

    @staticmethod
    def __name__() -> str:
        return "GalvoPointListMeta"


class MarkedPointSeriesMeta(_BrukerMeta):
    def __init__(self, root: ElementTree, factory: object):
        """
        Metadata object for MarkedPoints Protocols.

        loaded from an experiment or from a protocol template
        """
        self.marked_point_series = None

        super().__init__(root, factory)

    def __str__(self):
        return f"Marked point series meta containing {len(self.marked_point_series.marks)} marks"

    @staticmethod
    def __name__() -> str:
        return "Photostimulation Metadata"

    def _build_meta(self, root: ElementTree, factory: object) -> MarkedPointSeriesMeta:
        """
        Abstract method for building metadata object

        :param root:
        :param factory:
        :return:
        """
        self.marked_point_series = factory.constructor(root)

        marks = []
        for marked_point in root:
            points = tuple([factory.constructor(galvo_point) for galvo_point in marked_point])
            marked_point = factory.constructor(marked_point)
            marked_point.points = points
            marks.append(marked_point)

        self.marked_point_series.marks = tuple(marks)

    def _extra_actions(self) -> MarkedPointSeriesMeta:
        # for idx, roi in enumerate(self.rois):
        #    roi.coordinates = roi.generate_coordinates(self.image_width, self.image_height)
        #    roi.mask = roi.generate_mask(self.image_width, self.image_height)
        #    roi.stimulation_group = idx
        #    roi.vertices = roi.generate_hull_vertices()
        pass
