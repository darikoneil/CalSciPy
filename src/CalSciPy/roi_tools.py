from __future__ import annotations
from typing import Optional, Tuple, Union, Sequence, Iterable
from numbers import Number
from functools import partial
from abc import abstractmethod

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist


class ROI:
    """
    ROI object containing the relevant properties of some specific ROI.

    :param aspect_ratio: ratio of the short-to-long radius of the roi
    :param radius: approximate radius of the ROI
    :type radius: float
    :param shape: the shape of the image from which the roi was generated
    :type shape: Tuple[int, int]
    :param xpix: a 1D numpy array or Sequence indicating the x-pixels of the roi (column-wise)
    :type xpix: numpy.ndarray
    :param ypix: a 1D numpy array or Sequence indicating the y-pixels of the roi (row-wise)
    :type ypix: numpy.ndarray
    :ivar mask: approximate mask for the roi
    :type mask: Mask
    :ivar vertices: a 1D tuple indexing the convex hull of the roi (alternatively, may be considered an index of
        the pixels that form the boundaries of the roi
    :type vertices: Tuple[int]
    :ivar coordinates: the centroid of the roi (y, x) calculated using the shoelace approximation
    :type coordinates: Tuple[ float, float]
    :ivar adj_radii: The long and short radii of the roi (long, short)
    :type adj_radii: Tuple[float, float]
    :ivar xy: Nx2 array containing x,y coordinate pairs for the roi
    :type xy: numpy.ndarray
    :ivar xy_vert: Nx2 array containing the x,y coordinate pairs comprising the roi's convex hull approximation
    :type xy_vert: numpy.ndarray
    :ivar rc: Nx2 array containing the r,c coordinate pairs for the roi
    :type rc: numpy.ndarray
    :ivar rc_vert: Nx2 array containing the r,c coordinate pairs comprising the roi's convex hull approximation
    """
    def __init__(self,
                 xpix: Union[np.ndarray, Sequence[int]],
                 ypix: Union[np.ndarray, Sequence[int]],
                 aspect_ratio: float = 1.0,
                 plane: int = 0,
                 radius: float = 1.0,
                 shape: Tuple[float, float] = (512, 512)):
        """
        ROI object containing the relevant properties of some specific ROI for using in generating photostimulation
        protocols.

        :param xpix: a 1D numpy array or Sequence indicating the x-pixels of the roi (column-wise)
        :param ypix: a 1D numpy array or Sequence indicating the y-pixels of the roi (row-wise)
        :param aspect_ratio: ratio of the short-to-long radius of the roi
        :param plane: index of the imaging plane
        :param radius: approximate radius of the ROI
        :param shape: the shape of the image from which the roi was generated

        """
        self.aspect_ratio = aspect_ratio
        self.plane = plane
        self.radius = radius
        self.shape = shape
        self.xpix = xpix
        self.ypix = ypix

        self.vertices = identify_vertices(self.xpix, self.ypix)
        self.coordinates = calculate_centroid(self.xy_vert)[::-1]  # requires vertices!!!
        self.mask = Mask(self.coordinates, self.rc_vert, self.adj_radii, 0, self.shape)  # requires vertices!!!

    def __str__(self):
        return f"ROI centered at {tuple([round(val) for val in self.coordinates])}"

    @property
    def adj_radii(self) -> Tuple[float, float]:
        """
        The long and short radii of the roi (long, short)

        """
        short_radius = (2 * self.radius) / (self.aspect_ratio + 1)
        long_radius = self.aspect_ratio * short_radius
        return long_radius, short_radius

    @property
    def xy(self) -> np.ndarray:
        """
        Nx2 array containing x,y coordinate pairs for the roi

        """
        return np.vstack([self.xpix, self.ypix]).T

    @property
    def xy_vert(self) -> np.ndarray:
        """
        Nx2 array containing the x,y coordinate pairs comprising the roi's convex hull approximation

        """
        return self.xy[self.vertices, :]

    @property
    def rc(self) -> np.ndarray:
        """
        Nx2 array containing the r,c coordinate pairs for the roi

        """
        return np.vstack([self.ypix, self.xpix]).T

    @property
    def rc_vert(self) -> np.ndarray:
        """
        Nx2 array containing the r,c coordinate pairs comprising the roi's convex hull approximation

        """
        return self.rc[self.vertices, :]

    @staticmethod
    def __name__() -> str:
        return "ROI"

    def __repr__(self):
        return "ROI(" + "".join([f"{key}: {value} " for key, value in vars(self).items()]) + ")"


class ROIHandler:
    """
    Abstract object for generating reference images and ROI objects

    """

    @classmethod
    def load(cls: ROIHandler, *args, **kwargs) -> Tuple[dict, np.ndarray]:
        """
        Method that loads rois and generates reference image

        :returns: a dictionary in which each key is an integer indexing an ROI object and a reference image
        """

        rois_data_structure, reference_image_data_structure = cls.from_file(*args, **kwargs)

        reference_image = cls.generate_reference_image(reference_image_data_structure)

        rois = cls.import_rois(rois_data_structure, reference_image.shape)

        return rois, reference_image

    @staticmethod
    @abstractmethod
    def convert_one_roi(roi: Any, reference_shape: Tuple[int, int] = (512, 512)) -> ROI:
        """
        Abstract method for converting one roi in an ROI object

        :param roi: some sort of roi data structure
        :param reference_shape: the shape of the image containing the rois
        :return: a single ROI object
        """
        ...

    @staticmethod
    @abstractmethod
    def from_file(*args, **kwargs) -> Tuple[Any, Any]:
        """
        Abstract method to load the rois_data_structure and reference_image_data_structure from a file or folder

        :returns: data structures containing the rois and reference image
        """
        ...

    @staticmethod
    @abstractmethod
    def generate_reference_image(data_structure: Any) -> np.ndarray:
        """
        Abstract method to generate a reference image from some data structure

        :param data_structure: some data structure from which the reference image can be derived
        :return: reference image
        """
        ...

    @classmethod
    def import_rois(cls: ROIHandler,
                    rois: Union[Iterable, np.ndarray],
                    reference_shape: Tuple[int, int] = (512, 512)
                    ) -> dict:
        """
        Abstract method for importing rois

        :param rois: some sort of data structure iterating over all rois or a numpy array which will be converted to
            an Iterable
        :param reference_shape: the shape of the image containing the rois
        :return: dictionary containing in which the keys are integers indexing the roi and each roi is an ROI object
        """

        # Convert numpy array if provided
        if isinstance(rois, np.ndarray):
            rois = rois.tolist()

        converter = partial(cls.convert_one_roi, reference_shape=shape)
        return dict(enumerate([converter(element) for element in rois]))


class Suite2PHandler(ROIHandler):
    @staticmethod
    def convert_one_roi(roi: Any, reference_shape: Tuple[int, int] = (512, 512)) -> ROI:
        """
        Generates ROI from suite2p stat array

        :param roi: dictionary containing one suite2p roi
        :param reference_shape: shape of the reference image containing the roi
        :return: ROI instance for the roi
        """
        aspect_ratio = stat_element.get("aspect_ratio")
        radius = stat_element.get("radius")
        xpix = stat_element.get("xpix")[~stat_element.get("overlap")]
        ypix = stat_element.get("ypix")[~stat_element.get("overlap")]
        return ROI(aspect_ratio=aspect_ratio, radius=radius, shape=reference_shape, xpix=xpix, ypix=ypix)

    @staticmethod
    def from_file(folder: Path) -> Tuple[np.ndarray, dict]:
        """

        :param folder: folder containing suite2p data. The folder must contain the associated "stat.npy"
            & "ops.npy" files, though it is recommended the folder also contain the "iscell.npy" file.
        :returns: "stat" and "ops"
        """

        # append suite2p + plane if necessary
        if "suite2p" not in str(folder):
            folder = folder.joinpath("suite2p")

        if "plane" not in str(folder):
            folder = folder.joinpath("plane0")

        stat = np.load(folder.joinpath("stat.npy"), allow_pickle=True)

        # use only neuronal rois if iscell is provided
        try:
            iscell = np.load(folder.joinpath("iscell.npy"), allow_pickle=True)
        except FileNotFoundError:
            stat[:] = stat
        else:
            stat = stat[np.where(iscell[:, 0] == 1)[0]]

        ops = np.load(folder.joinpath("ops.npy"), allow_pickle=True).item()

        return stat, ops

    @staticmethod
    def generate_reference_image(data_structure: Any) -> np.ndarray:
        """
         Generates an appropriate reference image from suite2p ops dictionary

        :param data_structure: ops dictionary
        :return: reference image
        """

        true_shape = (ops.get("Ly"), ops.get("Lx"))

        # Load Vcorr as our reference image
        try:
            reference_image = ops.get("Vcorr")
            assert(reference_image is not None)
        except (KeyError, AssertionError):
            reference_image = np.ones(true_shape)

        # If motion correction cropped Vcorr, append minimum around edges
        if reference_image.shape != true_shape:
            true_reference_image = np.ones(true_shape) * np.min(reference_image)
            x_range = ops.get("xrange")
            y_range = ops.get("yrange")
            true_reference_image[y_range[0]: y_range[-1], x_range[0]:x_range[-1]] = reference_image
            return true_reference_image
        else:
            return reference_image


def calculate_bounding_radius(centroid: Sequence[Number, Number], vertices: np.ndarray) -> float:
    """
    Calculates the bounding radius of the roi, defined as the shortest distance between the centroid and the vertices of
    the approximate convex hull.

    :param centroid: centroid of the roi in row-column format (y, x)
    :param vertices: Nx2 array containing the vertices of the convex hull approximation
    :return: bounding radius
    """
    func_handle = partial(calculate_distance_from_centroid, centroid=centroid)
    return np.min([func_handle(point=vertices[point, :]) for point in range(vertices.shape[0])])


def calculate_centroid(roi: np.ndarray) -> Tuple[int, int]:
    """
    Calculates the centroid of a polygonal roi given an Nx2 numpy array containing the coordinates of the
    roi.The vertices of the roi's approximate convex hull are calculated (if necessary) and the centroid estimated
    from these vertices using the shoelace formula.

    :param roi: the full coordinates of one roi or the vertices of its approximate convex hull (x, y)
    :return: a tuple containing the centroid of the roi (x, y)
    """
    points = vertices.shape[0]

    center_x = 0
    center_y = 0
    sigma_signed_area = 0

    for pt in range(points):
        if pt < points - 1:
            trapezoid_area = (vertices[pt, 0] * vertices[pt + 1, 1]) - (vertices[pt + 1, 0] * vertices[pt, 1])
            sigma_signed_area += trapezoid_area
            center_x += (vertices[pt, 0] + vertices[pt + 1, 0]) * trapezoid_area
            center_y += (vertices[pt, 1] + vertices[pt + 1, 1]) * trapezoid_area
        else:
            trapezoid_area = (vertices[pt, 0] * vertices[0, 1]) - (vertices[0, 0] * vertices[pt, 1])
            sigma_signed_area += trapezoid_area
            center_x += (vertices[pt, 0] + vertices[0, 0]) * trapezoid_area
            center_y += (vertices[pt, 1] + vertices[0, 1]) * trapezoid_area

    signed_area = abs(sigma_signed_area) / 2
    center_x /= (6 * signed_area)
    center_y /= (6 * signed_area)

    return center_x, center_y


def calculate_distance_from_centroid(centroid: Sequence[Number, Number], point: np.ndarray) -> float:
    """
    Calculates the euclidean distance between the centroid and a specific point by wrapping scipy's
    pdist implementation

    :param centroid: the center of the roi in row-column form (y, x)
    :param point: a specific point row-column form (y, x)
    :return: the distance between the center and the point
    """
    coordinate_pair = np.empty((2, 2))
    coordinate_pair[0, :] = centroid
    coordinate_pair[1, :] = point
    return pdist(coordinate_pair, metric="euclidean")


def calculate_mask(centroid: Sequence[Number, Number],
                   radii: Union[Number, Sequence[Number, Number]],
                   shape: Union[Number, Sequence[Number, Number]] = None,
                   theta: Number = None
                   ) -> np.ndarray:
    """
    Calculates an approximate mask for an elliptical roi at center with radii at theta with respect to the y-axis
    and constrained to lie within the dimensions imposed by shape

    :param centroid: centroid of the roi in row-column form (y, x)
    :param radii: radius of the roi. can provide one radius for a symmetrical roi or a long and short radius.
    :param shape: dimensions of the image the roi lies within. If only one value is provided it is considered
        symmetrical.
    :param theta: angle of the long-radius with respect to the y-axis
    :return: photostimulation mask
    """

    if theta is not None:
        raise NotImplementedError("Implementation Pending")

    # ensure center is numpy array
    centroid = np.asarray(centroid)

    # make sure radii contains both x & y directions
    try:
        assert (len(radii) == 2)
    except TypeError:
        radii = np.asarray([radii, radii])
    except AssertionError:
        radii = np.asarray([*radii, *radii])

    # generate a rectangle that bounds our mask (upper left, lower right)
    bounding_rect = np.vstack([
        np.ceil(centroid - radii).astype(int),
        np.floor(centroid + radii).astype(int),
    ])

    # constrain to within the shape of the image, if necessary
    if shape is not None:
        bounding_rect[:, 0] = bounding_rect[:, 0].clip(0, shape[0] - 1)
        bounding_rect[:, 1] = bounding_rect[:, 1].clip(0, shape[-1] - 1)

    # adjust center
    centroid -= bounding_rect[0, :]

    bounding = bounding_rect[1, :] - bounding_rect[0, :] + 1

    y_grid, x_grid = np.ogrid[0:float(bounding[0]), 0:float(bounding[1])]

    y, x = centroid
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


def identify_vertices(roi: Union[np.ndarray, Sequence[int]],
                      ypix: Optional[Union[np.ndarray, Sequence[int]]] = None
                      ) -> Tuple[int]:
    """
    Calculate the index of points comprising the approximate convex hull the polygon. Wraps scipy's ConvexHull,
    which is itself an implementation of QtHull


    :param roi: a 1D numpy array or Sequence indicating the x-pixels of the roi
        or an Nx2 numpy array containing the coordinates of the roi (x, y)
    :param ypix: a 1D numpy array or Sequence indicating the y-pixels of the roi if and only if roi is 1D
    :return:  a 1D tuple indexing the vertices of the approximate convex hull of the roi
        (alternatively, may be considered an index of the pixels that form the boundaries of the roi)
    """
    if ypix is not None:
        roi = np.vstack([roi, ypix]).T

    hull = ConvexHull(roi)

    return hull.vertices
