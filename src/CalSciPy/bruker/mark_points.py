from __future__ import annotations
from xml.etree import ElementTree
from .bruker_meta_objects import BrukerMeta
from collections import ChainMap
from PPVD.style import TerminalStyle
import numpy as np
from scipy.spatial import ConvexHull

import matplotlib


class PhotostimulationMeta(BrukerMeta):
    def __init__(self, root: ElementTree, factory: object, width: int = 512, height: int = 512):
        """
        Metadata object for Photostimulation / MarkedPoints Protocols.

        Can either be loaded from an experiment OR built to generate new experiments
        """
        self.sequence = None
        self.groups = []
        self.rois = []
        self.image_width = width
        self.image_height = height

        super().__init__(root, factory)
        return

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
            self.rois.append(ROI(*roi))

    def _extra_actions(self) -> BrukerMeta:
        for idx, roi in enumerate(self.rois):
            roi.coordinates = roi.generate_coordinates(self.image_width, self.image_height)
            roi.mask = roi.generate_mask(self.image_width, self.image_height)
            roi.stimulation_group = idx
            roi.vertices = roi.generate_hull_vertices()

    def generate_protocol(self, path: str) -> None:
        """
        Generates a protocol for the metadata to be imported into prairieview

        :param path: path to write protocol
        :type path: str or pathlib.Path
        :rtype: None
        """
        pass


class Group:
    def __init__(self, ):
        self.parameters = []
        self.rois = []


class ROI:
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

    def _pull_parameters_to_upper_level(self):
        params = []
        for parameter_set in self._map.maps:
            params.append(parameter_set.__dict__)
        self.parameters = dict(ChainMap(*params))
        self.index = self.parameters.get("index")

    def generate_coordinates(self, width: int, height: int) -> Tuple[float, float]:
        """
        Converts the normalized coordinates to image coordinates

        :param width: width of image
        :type width: int
        :param height: height of image
        :type height: int
        :return:  x,y coordinates
        :rtype: tuple[float, float]
        """
        return self.parameters.get("x") * width, self.parameters.get("y") * height

    def generate_mask(self, width: int, height: int) -> Tuple[Tuple[int, int]]:
        """
        Converts spiral center & radii to a coordinate mask

        :param width: width of image
        :type width: int
        :param height: height of image
        :type height: int
        :return: coordinate mask (y, x)
        """
        # reverse to get y, x order (row, column)
        center = np.asarray(self.coordinates[::-1])
        # get radii
        radii = (
            self.parameters.get("spiral_height") * height / 2, self.parameters.get("spiral_width") * width / 2, )
        # calculate mask
        return generate_photostimulation_mask(center, radii, (height, width))

    def generate_hull_vertices(self) -> Tuple[Tuple[int, int]]:
        """
        Identifies the vertices of the Convex-Hull approximation

        :return: vertices (Nx2)
        """
        pts = np.vstack([self.mask[0], self.mask[1]]).T
        hull = ConvexHull(pts)
        y, x = pts[hull.vertices, 0], pts[hull.vertices, 1]
        return [(y, x) for y, x in zip(y, x)]

    def __str__(self):
        """
        Prints the roi and each of its parameters, their values. It skips over the underlying chain map & the mask

        :rtype: str
        """
        string_to_print = ""
        string_to_print += f"\n{TerminalStyle.YELLOW}{TerminalStyle.BOLD}{TerminalStyle.UNDERLINE}" \
                           f"{self.__name__()}{TerminalStyle.RESET}\n"
        for key, value in vars(self).items():
            if isinstance(value, dict):
                string_to_print += f"{TerminalStyle.YELLOW}{TerminalStyle.BOLD}{key}{TerminalStyle.RESET}:"
                for nested_key in vars(self).get(key):
                    string_to_print += f"\n\t{TerminalStyle.YELLOW}{nested_key}: {TerminalStyle.RESET}" \
                                       f"{vars(self).get(key).get(nested_key)}"
                string_to_print += "\n"
            elif isinstance(value, ChainMap):
                pass
            elif isinstance(value, np.ndarray):
                pass
            else:
                string_to_print += f"{TerminalStyle.YELLOW}{TerminalStyle.BOLD}{key}: " \
                                   f"{TerminalStyle.RESET}{vars(self).get(key)}\n"
        return string_to_print

    def __repr__(self):
        return "ROI(" + "".join([f"{key}:{getattr(self, key)}, " for key in vars(self)]) + ")"

    @staticmethod
    def __name__():
        return "ROI"


def generate_photostimulation_mask(center: Sequence[Number, Number],
                                   radii: Union[Number, Sequence[Number]],
                                   shape: Union[Number, Sequence[Number, Number]] = None) -> np.ndarray:

    # ensure center is numpy array
    center = np.asarray(center)

    # make sure radii contains both x & y directions
    try:
        assert(len(radii) == 2)
    except TypeError:
        radii = np.asarray([radii, radii])
    except AssertionError:
        radii = np.asarray([*radii, *radii])

    # generate a rectangle that bounds our mask (upper left, lower right)
    bounding_rect = np.vstack([
        np.ceil(center - radii).astype(int),
        np.floor(center + radii).astype(int),
    ])

    # constrain to within the shape of the image, if necessary
    if shape is not None:
        bounding_rect[:, 0] = bounding_rect[:, 0].clip(0, shape[0] - 1)
        bounding_rect[:, 1] = bounding_rect[:, 1].clip(0, shape[-1] - 1)

    # adjust center
    center -= bounding_rect[0, :]

    bounding = bounding_rect[1, :] - bounding_rect[0, :] + 1

    y_grid, x_grid = np.ogrid[0:float(bounding[0]), 0:float(bounding[1])]

    y, x = center
    r_rad, c_rad = radii
    r, c = (y_grid - y), (x_grid - x)
    distances = (r / r_rad)**2 + (c / c_rad)**2

    yy, xx = np.nonzero(distances < 1)

    yy += bounding_rect[0, 0]
    xx += bounding_rect[0, 1]

    if shape is not None:
        yy.clip(0, shape[0] - 1)
        xx.clip(0, shape[-1] - 1)

    return yy, xx
