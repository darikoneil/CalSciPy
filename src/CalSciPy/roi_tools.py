from __future__ import annotations
from typing import Optional, Tuple, Union, Sequence, Iterable, Any, Mapping
from numbers import Number
from collections import ChainMap
from functools import partial, cached_property
from abc import abstractmethod
from pathlib import Path
from operator import eq, le

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist

from CalSciPy._backports import PatternMatching


"""
Object-oriented approach to organizing ROI-data
"""


class _ROIBase:
    """
    An abstract ROI object containing the base characteristics & properties of an ROI. Technically,
    the only abstract method is __name__. Therefore, it isn't *really* abstract, but it is not meant
    to be instanced; it contains the abstract method for protection. Note that the properties
    are only calculated once.

    """
    def __init__(self,
                 pixels: Union[NDArray[int], Sequence[int]],
                 ypixels: Union[NDArray[int], Sequence[int]],
                 reference_shape: Sequence[float, float] = (512, 512),
                 plane: Optional[int] = None,
                 properties: Optional[Mapping] = None,
                 zpix: Optional[Union[NDArray[int], Sequence[int]]] = None,
                 **kwargs
                 ):
        """
        An abstract ROI object containing the base characteristics & properties of an ROI. Technically,
        the only abstract method is __name__. Therefore, it isn't *really* abstract, and it is not meant
        to be instanced; it contains the abstract method for protection. Note that the properties
        are only calculated once.

        :param pixels: Nx2 array of x and y-pixel pairs **strictly** in rc form.
            If this argument is one-dimensional, it will be considered as an ordered sequence of x-pixels.
            The matching y-pixels must be then be provided as an additional argument.

        :param ypixels: The y-pixels of the roi if and only if the first argument is one-dimensional.

        :param reference_shape: the shape of the reference image from which the roi was generated

        :param plane: index of the imaging plane (if multi-plane)

        :param properties: optional properties to include

        :param zpix: The z-pixels of the roi (if volumetric)

        """
        #: NDArray[int]: the x-pixels of the roi (column-wise)
        self.xpix = None
        #: NDArray[int]: the y-pixels of the roi (row-wise)
        self.ypix = None

        # put pixels in proper format
        self.ypix, self.xpix = _validate_pixels(pixels, ypixels)

        #: Tuple[float, float, ...]: the shape of the image from which the roi was generated
        self.reference_shape = tuple(reference_shape)

        #: Optional[int]: index of the imaging plane (if multiplane)
        self.plane = plane
        #: Optional[NDArray[int]]: z-pixels of the roi if volumetric
        self.zpix = np.asarray(zpix)
        
        # cover non-implemented optionals
        if plane is not None or zpix is not None:
            raise NotImplementedError

        #: ChainMap: a mapping of properties containing any relevant information about the ROI
        self.properties = ChainMap(kwargs, properties)
        # user-defined, using chainmap is O(N) worst-case while dict construction / update
        # is O(NM) worst-case. Likely to see use in situations with thousands of constructions
        # with unknown number of parameters, so this is relevant

        #: Tuple[int, ...]: a tuple indexing the vertices of the approximate convex hull of the roi
        self.vertices = identify_vertices(self.xpix, self.ypix)

        #: Tuple[float, float, ...]: the centroid of the roi
        self.centroid = calculate_centroid(self.xy_vert)[::-1]  # requires vertices!!!

        #: float: the radius of the ROI
        self.radius = calculate_radius(self.centroid, self.rc_vert, method="mean")  # requires vertices + centroid!!!

    def __str__(self):
        return f"ROI centered at {tuple([round(val) for val in self.centroid])}"

    @cached_property
    def mask(self) -> NDArray[bool]:
        """
        Boolean mask with the dimensions of the reference image indicating the pixels of the ROI

        """
        mask = np.zeros(self.reference_shape, dtype=np.bool_)
        for pair in range(self.rc.shape[0]):
            y, x = self.rc[pair, :]
            mask[y, x] = True
        return mask

    @cached_property
    def xy(self) -> NDArray[int]:
        """
        Nx2 array containing x,y coordinate pairs for the roi

        """
        return np.vstack([self.xpix, self.ypix]).T

    @cached_property
    def xy_vert(self) -> NDArray[int]:
        """
        Nx2 array containing the x,y coordinate pairs comprising the roi's approximate convex hull

        """
        return self.xy[self.vertices, :]

    @cached_property
    def rc(self) -> NDArray[int]:
        """
        Nx2 array containing the r,c coordinate pairs for the roi

        """
        return np.vstack([self.ypix, self.xpix]).T

    @cached_property
    def rc_vert(self) -> NDArray[int]:
        """
        Nx2 array containing the r,c coordinate pairs comprising the roi's approximate convex hull

        """
        return self.rc[self.vertices, :]

    @staticmethod
    @abstractmethod
    def __name__() -> str:
        ...

    def __repr__(self):
        return "ROI(" + "".join([f"{key}: {value}, " for key, value in vars(self).items()]) + ")"


class ROI(_ROIBase):
    """
    An ROI object containing the base characteristics & properties of an ROI. Note that the properties are only
    calculated once.

    :param xpix: a 1D numpy array or Sequence indicating the x-pixels of the roi (column-wise)
    :param ypix: a 1D numpy array or Sequence indicating the y-pixels of the roi (row-wise)
    :param reference_shape: the shape of the image from which the roi was generated
    :param plane: index of the imaging plane (if multi-plane)
    :param properties: optional properties to include
    :param zpix: a 1D numpy array or Sequence indicating the z-pixels of the roi (if volumetric)

    """
    def __init__(self,
                 xpix: Union[NDArray[int], Sequence[int]],
                 ypix: Union[NDArray[int], Sequence[int]],
                 reference_shape: Tuple[float, float] = (512, 512),
                 method: str = "literal",
                 plane: Optional[int] = None,
                 properties: Optional[Mapping] = None,
                 zpix: Optional[Union[NDArray[int], Sequence[int]]] = None,
                 **kwargs):

        # initialize new attr
        #: str: method used to generate roi approximation
        self._method = None

        #: ApproximateROI: an approximation of the roi
        self.approximation = None

        # initialize parent attr
        super().__init__(xpix, ypix, reference_shape, plane, properties, zpix, **kwargs)

    @staticmethod
    def __name__() -> str:
        return "ROI"

    @property
    def approx_method(self) -> str:
        """
        Method used for approximating the roi

        """
        return self._method

    @approx_method.setter
    def approx_method(self, method: str = "literal") -> ROI:
        if method != self._method:
            self.approximation = ApproximateROI(self, method)
            self._method = method


class ApproximateROI(_ROIBase):
    def __init__(self,
                 roi: ROI,
                 method: str = "literal"):

        # initialize new attr
        self._method = method

        # initialize parent attr
        super().__init__(
            *self.__pre_init__(roi, method)
        )

    def __str__(self):
        return f"ROI approximation centered at {self.centroid} with radius {self.radius}"

    @property
    def method(self) -> str:
        return self._method

    @staticmethod
    def __name__() -> str:
        return "ROI Approximation"

    @classmethod
    def __pre_init__(cls: ApproximateROI,
                     roi: ROI,
                     method: str = "literal"
                     ) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
        if method == "literal":
            xpix = roi.xpix
            ypix = roi.ypix
        else:
            radius = calculate_radius(roi.centroid, roi.rc_vert, method=method)
            ypix, xpix = calculate_mask(roi.centroid, radius, roi.reference_shape)
        return xpix, ypix, roi.reference_shape

    def __repr__(self):
        return "ROI Approximation(" + "".join([f"{key}: {value} " for key, value in vars(self).items()]) + ")"


class ROIHandler:
    """
    Abstract object for generating reference images and ROI objects

    """

    @staticmethod
    @abstractmethod
    def convert_one_roi(roi: Any, reference_shape: Tuple[int, int] = (512, 512)) -> ROI:
        """
        Abstract method for converting one roi in an ROI object

        :param roi: some sort of roi data structure
        :param reference_shape: the reference_shape of the image containing the rois
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
        :param reference_shape: the reference_shape of the image containing the rois
        :return: dictionary containing in which the keys are integers indexing the roi and each roi is an ROI object
        """

        # Convert numpy array if provided
        if isinstance(rois, np.ndarray):
            rois = rois.tolist()

        converter = partial(cls.convert_one_roi, reference_shape=reference_shape)
        return dict(enumerate([converter(element) for element in rois]))

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


class Suite2PHandler(ROIHandler):
    @staticmethod
    def convert_one_roi(roi: Any, reference_shape: Tuple[int, int] = (512, 512)) -> ROI:
        """
        Generates ROI from suite2p stat array

        :param roi: dictionary containing one suite2p roi
        :param reference_shape: reference_shape of the reference image containing the roi
        :return: ROI instance for the roi
        """
        aspect_ratio = roi.get("aspect_ratio")
        radius = roi.get("radius")
        xpix = roi.get("xpix")[~roi.get("overlap")]
        ypix = roi.get("ypix")[~roi.get("overlap")]
        return ROI(aspect_ratio=aspect_ratio, radius=radius, reference_shape=reference_shape, xpix=xpix, ypix=ypix)

    @staticmethod
    def from_file(folder: Path, *args, **kwargs) -> Tuple[np.ndarray, dict]:  # noqa: U100
        """

        :keyword folder: folder containing suite2p data. The folder must contain the associated "stat.npy"
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

        true_shape = (data_structure.get("Ly"), data_structure.get("Lx"))

        # Load Vcorr as our reference image
        try:
            reference_image = data_structure.get("Vcorr")
            assert (reference_image is not None)
        except (KeyError, AssertionError):
            reference_image = np.ones(true_shape)

        # If motion correction cropped Vcorr, append minimum around edges
        if reference_image.shape != true_shape:
            true_reference_image = np.ones(true_shape) * np.min(reference_image)
            x_range = data_structure.get("xrange")
            y_range = data_structure.get("yrange")
            true_reference_image[y_range[0]: y_range[-1], x_range[0]:x_range[-1]] = reference_image
            return true_reference_image
        else:
            return reference_image


def calculate_radius(centroid: Sequence[Number, Number],
                     vertices_coordinates: NDArray[int],
                     method: str = "mean"
                     ) -> Union[float, Tuple[Tuple[float, float], float]]:
    """
    Calculates the radius of the roi using one of the following methods:
        #   "mean": a symmetrical radius calculated as the average distance between the centroid and the vertices of
            the approximate convex hull
        #   "bound": a symmetrical radius calculated as the minimum distance between the centroid and the vertices of
            the approximate convex hull
        #   "unbound": a symmetrical radius calculated as the maximal distance between the centroid and the vertices
            of the approximate convex hull
        #   "ellipse" : an asymmetrical set of radii whose major-axis radius forms the angle theta with respect to the
            y-axis of the reference image

    :param centroid: centroid of the roi in row-column format (y, x)

    :param vertices_coordinates: Nx2 array containing the pixels that form the vertices of the approximate convex hull

    :param method: method to use when calculating radius ("mean", "bound", "unbound", "ellipse")

    :returns: the radius of the roi for symmetrical radi;
        the major & minor radii and the value of theta for asymmetrical radii
    """

    center = np.asarray(centroid)
    center = np.reshape(center, (1, 2))
    radii = cdist(center, vertices_coordinates)

    if method == "mean":
        return np.mean(radii)
    elif method == "bound":
        return np.min(radii)
    elif method == "unbound":
        return np.max(radii)
    # secret for debugs
    elif method == "all":
        return radii
    else:
        raise NotImplementedError(f"Request method {method} is not supported")


def calculate_centroid(pixels: Union[NDArray[int], Sequence[int]],
                       ypixels: Optional[Union[NDArray[int], Sequence[int, ...]]] = None
                       ) -> Tuple[float, float]:
    """
    Calculates the centroid of a polygonal roi .The vertices of the roi's approximate convex hull are calculated
    (if necessary) and the centroid estimated from these vertices using the shoelace formula.

    :param pixels: Nx2 array of x and y-pixel pairs in xy or rc form. If this argument is one-dimensional,
        it will be considered as an ordered sequence of x-pixels. The matching y-pixels must be then be provided
        as an additional argument.

    :param ypixels: The y-pixels of the roi if and only if the first argument is one-dimensional.

    :returns: a tuple containing the centroid of the roi. Whether the centroid is in xy or rc form is dependent on the
        form of the arguments
    """
    # if ypix is provided and xpix is one dimensional
    # we have to do it weird this way because many users will pass a 2D array that is of shape (N, 1) rather than (N, )
    ypixels, pixels = _validate_pixels(pixels, ypixels)

    # we need RC
    pixels = np.vstack([ypixels, pixels]).T

    # calculate convex hull (if necessary)
    vertices = pixels[identify_vertices(pixels), :]

    center_x = 0
    center_y = 0
    sigma_signed_area = 0

    points = vertices.shape[0]
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


def calculate_mask(centroid: Sequence[Number, Number],
                   radii: Union[Number, Sequence[Number, Number]],
                   reference_shape: Union[Number, Sequence[Number, Number]] = None,
                   theta: Number = None
                   ) -> NDArray[bool]:
    """
    Calculates a boolean mask for an elliptical roi constrained to lie within the dimensions imposed by the reference
    shape.The major-axis is considered to have angle theta with respect to the y-axis of the reference image.

    :param centroid: centroid of the roi in row-column form (y, x)

    :param radii: radius of the roi. Only one radius is required if the roi is symmetrical (i.e., circular).
        For an elliptical roi both a long and short radius can be provided.
    
    :param reference_shape: dimensions of the reference image the roi lies within. If only one value is provided
        it is considered symmetrical.

    :param theta: angle of the long-radius with respect to the y-axis of the reference image
    
    :returns: a boolean mask identifying which pixels contain the roi within the reference image
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

    # constrain to within the reference_shape of the image, if necessary
    if shape is not None:
        bounding_rect[:, 0] = bounding_rect[:, 0].clip(0, shape[0] - 1)
        bounding_rect[:, 1] = bounding_rect[:, 1].clip(0, shape[-1] - 1)

    # adjust center
    centroid -= bounding_rect[0, :]

    # bounding  shape
    bounding = bounding_rect[1, :] - bounding_rect[0, :] + 1
    y_grid, x_grid = np.ogrid[0:float(bounding[0]), 0:float(bounding[1])]

    # origin
    y, x = centroid
    r_rad, c_rad = radii

    #         ((x * cos(alpha) + y * sin(alpha)) / x_radius) ** 2 +
    #         ((x * sin(alpha) - y * cos(alpha)) / y_radius) ** 2 = 1

    r, c = (y_grid - y), (x_grid - x)
    distances = (r / r_rad)**2 + (c / c_rad)**2

    # collect
    yy, xx = np.nonzero(distances < 1)

    # adj bounds
    yy += bounding_rect[0, 0]
    xx += bounding_rect[0, 1]

    # constrain to within the reference_shape of the image, if necessary
    if shape is not None:
        yy.clip(0, shape[0] - 1)
        xx.clip(0, shape[-1] - 1)

    return yy, xx


def identify_vertices(pixels: Union[NDArray[int], Sequence[int]],
                      ypixels: Optional[Union[NDArray[int], Sequence[int]]] = None
                      ) -> Tuple[int, ...]:
    """
    Identifies the points of a given polygon which form the vertices of the approximate convex hull. This function wraps 

    :class:`scipy.spatial.ConvexHull`, which is an ultimately a wrapper for `QHull <https://www.qhull.org>`_. It's a
        fast and easy alternative to actually determining the "true" boundary vertices given the assumption that
        cellular ROIs are convex (i.e., cellular rois ought to be roughly elliptical).

    :param pixels: Nx2 array of x and y-pixel pairs in xy or rc form. If this argument is one-dimensional,
        it will be considered as an ordered sequence of x-pixels. The matching y-pixels must be then be provided
        as an additional argument.

    :param ypixels: The y-pixels of the roi if and only if the first argument is one-dimensional.
    
    :returns: A tuple indexing which points form the vertices of the approximate convex hull.
        It may alternatively be considered an index of the smallest set of pixels that are able to demarcate the
        boundaries of the roi, though this only holds if the polygon doesn't have any concave portions. 
    """

    # if not numpy array, convert
    pixels = np.asarray(pixels)

    ypixels, pixels = _validate_pixels(pixels, ypixels)
    pixels = np.vstack([ypixels, pixels]).T

    # approximate convex hull
    hull = ConvexHull(pixels)

    # return the vertices
    return hull.vertices


def _validate_pixels(pixels: Union[NDArray[int], Sequence[int]],
                     ypixels: Optional[Union[NDArray[int], Sequence[int]]]
                     ) -> Tuple[NDArray[int], NDArray[int]]:

    pixels = np.asarray(pixels)
    if ypixels is None:
        assert(sum(pixels.shape) >= max(pixels.shape) + 1)
        return pixels[:, 0], pixels[:, 1]
    else:
        ypixels = np.asarray(ypixels)
        assert(ypixels.shape == pixels.shape)
        return ypixels, pixels
