from __future__ import annotations
from typing import Tuple
from abc import abstractmethod
from xml.etree import ElementTree
from collections import ChainMap

import numpy as np
from scipy.spatial import ConvexHull

from ...color_scheme import TERM_SCHEME
from ...roi_tools import calculate_mask


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
    def __init__(self, root: ElementTree, factory: object, width: int = 512, height: int = 512):
        """
        Metadata object for a galvo point list saved from mark points in pyprairieview

        """
        self.stuff = []
        self.width = width
        self.height = height

        super().__init__(root, factory)

    def _build_meta(self, root: ElementTree, factory: object) -> _BrukerMeta:
        children = [child for idx, child in enumerate(root) if idx >= 25]
        child = children[0]
        # beep
        # factory.constructor(child)

    def _extra_actions(self, *args, **kwargs) -> _BrukerMeta:
        return 0

    @staticmethod
    def __name__() -> str:
        return "GalvoPointListMeta"


class PhotostimulationMeta(_BrukerMeta):
    def __init__(self, root: ElementTree, factory: object, width: int = 512, height: int = 512):
        """
        Metadata object for Photostimulation / MarkedPoints Protocols.

        loaded from an experiment
        """
        self.sequence = None
        self.groups = []
        self.rois = []
        self.image_width = width
        self.image_height = height

        super().__init__(root, factory)

    def __str__(self):
        return f"Photostimulation metadata containing {len(self.rois)} ROIs within " \
               f"{self.image_width} x {self.image_height} reference image (x, y)"

    @staticmethod
    def __name__() -> str:
        return "Photostimulation Metadata"

    def _build_meta(self, root: ElementTree, factory: object) -> PhotostimulationMeta:
        """
        Abstract method for building metadata object

        :param root:
        :param factory:
        :return:
        """
        self.sequence = factory.constructor(root)
        self._roi_constructor(root, factory)

    def _extra_actions(self) -> PhotostimulationMeta:
        for idx, roi in enumerate(self.rois):
            roi.coordinates = roi.generate_coordinates(self.image_width, self.image_height)
            roi.mask = roi.generate_mask(self.image_width, self.image_height)
            roi.stimulation_group = idx
            roi.vertices = roi.generate_hull_vertices()

    def _roi_constructor(self, root: ElementTree, factory: object) -> PhotostimulationMeta:
        """
        Constructs each ROI

        :param root:
        :param factory:
        :return:
        """
        # self.groups = [Group(group) for group in root]
        for roi_ in root:
            roi = []
            for child in roi_.iter():
                roi.append(factory.constructor(child))
            self.rois.append(ROIMeta(*roi))


class GroupMeta:
    def __init__(self, ):
        self.parameters = []
        self.rois = []


class ROIMeta:
    """
    ROI Object

    """
    def __init__(self, *args):

        self.coordinates = None

        self.mask = None

        self.vertices = None

        self.index = None

        self.stimulation_group = None

        self.parameters = {}

        self._map = ChainMap(*args)

        self._pull_parameters_to_upper_level()

    def __str__(self):
        """
        Prints the roi and each of its parameters, their values. It skips over the underlying chain map & the mask

        :rtype: str
        """
        string_to_print = TERM_SCHEME(f"{self.__name__()}\n", "header")
        for key, value in vars(self).items():
            if isinstance(value, dict):
                string_to_print += TERM_SCHEME(f"{key}", "emphasis")
                string_to_print += ":"
                for nested_key in vars(self).get(key):
                    string_to_print += f"\n\t{nested_key}: {vars(self).get(key).get(nested_key)}"
                string_to_print += "\n"
            elif isinstance(value, ChainMap):
                pass
            elif isinstance(value, np.ndarray):
                pass
            else:
                string_to_print += TERM_SCHEME(f"{key}", "emphasis")
                string_to_print += f": {vars(self).get(key)}\n"
        return string_to_print

    @staticmethod
    def __name__() -> str:
        return "ROI"

    def generate_coordinates(self, width: int, height: int) -> Tuple[float, float]:
        """
        Converts the normalized coordinates to image coordinates

        :param width: width of image
        :param height: height of image
        :return:  x,y coordinates
        """
        return self.parameters.get("x") * width, self.parameters.get("y") * height

    def generate_hull_vertices(self) -> Tuple[Tuple[int, int]]:
        """
        Identifies the vertices of the Convex-Hull approximation

        :return: vertices (Nx2)
        """
        pts = np.vstack([self.mask[0], self.mask[1]]).T
        hull = ConvexHull(pts)
        y, x = pts[hull.vertices, 0], pts[hull.vertices, 1]
        return [(y, x) for y, x in zip(y, x)]

    def generate_mask(self, width: int, height: int) -> Tuple[Tuple[int, int]]:
        """
        Converts spiral center & radii to a coordinate mask

        :param width: width of image
        :param height: height of image
        :return: coordinate mask (y, x)
        """
        # reverse to get y, x order (row, column)
        center = np.asarray(self.coordinates[::-1])
        # get radii
        radii = (
            self.parameters.get("spiral_height") * height / 2, self.parameters.get("spiral_width") * width / 2, )
        # calculate mask
        return calculate_mask(center, radii, (height, width))

    def _pull_parameters_to_upper_level(self) -> ROIMeta:
        params = []
        for parameter_set in self._map.maps:
            params.append(parameter_set.__dict__)
        self.parameters = dict(ChainMap(*params))
        self.index = self.parameters.get("index")

    def __repr__(self):
        return "ROI(" + "".join([f"{key}:{getattr(self, key)}, " for key in vars(self)]) + ")"
