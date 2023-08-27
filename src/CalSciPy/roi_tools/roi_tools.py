from __future__ import annotations
from typing import Optional, Tuple, Union, Sequence, Iterable, Any, Mapping
from numbers import Number
from collections import ChainMap
from functools import partial, cached_property
from abc import abstractmethod, ABCMeta

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist


"""
Object-oriented approach to organizing ROI-data
"""


class _ROIBase(metaclass=ABCMeta):
    """
    An abstract ROI object containing the base characteristics & properties of an ROI. Technically,
    the only abstract method is __name__. Therefore, it isn't *really* abstract, but it is not meant
    to be instanced; it contains the abstract method for protection. Note that the properties
    are only calculated once.

    """
    def __init__(self,
                 pixels: Union[NDArray[int], Sequence[int]],
                 y_pixels: Union[NDArray[int], Sequence[int]] = None,
                 reference_shape: Sequence[float, float] = (512, 512),
                 plane: Optional[int] = None,
                 properties: Optional[Mapping] = None,
                 z_pixels: Optional[Union[NDArray[int], Sequence[int]]] = None,
                 **kwargs
                 ):
        """
        An abstract ROI object containing the base characteristics & properties of an ROI. Technically,
        the only abstract method is __name__. Therefore, it isn't *really* abstract, and it is not meant
        to be instanced; it contains the abstract method for protection. Note that the properties
        are only calculated once.

        :param pixels: Nx2 array of x and y-pixel pairs in xy or rc form. If this argument is one-dimensional,
            it will be considered as an ordered sequence of x-pixels. The matching y-pixels must be then be provided
            as an additional argument.

        :type pixels: :class:`Union <typing.Union>` [ :class:`ndarray <numpy.ndarray>` [ :class:`Any <typing.Any>` ,
            :class:`dtype <numpy.dtype>` [ :class:`int` ], :class:`Sequence <typing.Sequence>` [ :class:`int` ]]

        :param y_pixels: The y-pixels of the roi if and only if the first argument is one-dimensional.

        :type y_pixels: :class:`Optional <typing.Optional>` [ :class:`Union <typing.Union>`
            [ :class:`ndarray <numpy.ndarray>` [ :class:`Any <typing.Any>` , :class:`dtype <numpy.dtype>` [ :class:`int`
            ], :class:`Sequence <typing.Sequence>` [ :class:`int` ]]], default: None

        :param reference_shape: The shape of the reference image from which the roi was generated

        :param plane: Index of the imaging plane (if multi-plane)

        :param properties: Optional properties to include

        :param z_pixels: The z-pixels of the roi (if volumetric)

        :type z_pixels: :class:`Optional <typing.Optional>` [ :class:`Union <typing.Union>`
            [ :class:`ndarray <numpy.ndarray>` [ :class:`Any <typing.Any>` , :class:`dtype <numpy.dtype>`
            [ :class:`int` ]], :class:`Sequence <typing.Sequence>` [ :class:`int` ]]], default: None
        """
        # true x pixels
        self._x_pixels = None
        # true y pixels
        self._y_pixels = None
        # flag if pixels already set
        self._x_set = False
        self._y_set = False

        # put pixels in proper format
        self.y_pixels, self.x_pixels = _validate_pixels(pixels, y_pixels)

        #: :class:`Tuple <typing.Tuple>` [ :class:`float` , :class:`float` , ... ]: the shape of the image from which the roi was generated
        self.reference_shape = tuple(reference_shape)

        #: :class:`Optional <typing.Optional>` [ :class:`int` ]: index of the imaging plane (if multiplane)
        self.plane = plane

        #: :class:`Optional <typing.Optional>` [ :class:`ndarray <numpy.ndarray>` [ :class:`Any <typing.Any>` , :class:`dtype <numpy.dtype>` [ :class:`int` ]]]: z-pixels of the roi if volumetric
        self.z_pixels = z_pixels

        # cover non-implemented optionals
        if plane is not None or z_pixels is not None:
            raise NotImplementedError

        #: :class:`ChainMap <collections.ChainMap>`: a mapping of properties containing any relevant information about the ROI
        self.properties = ChainMap(kwargs, properties)
        # user-defined, using chainmap is O(N) worst-case while dict construction / update
        # is O(NM) worst-case. Likely to see use in situations with thousands of constructions
        # with unknown number of parameters, so this is relevant

        #: :class:`Tuple <typing.Tuple>` [ :class:`int` , ... ]: a tuple indexing the vertices of the approximate convex hull of the roi
        self.vertices = identify_vertices(self.x_pixels, self.y_pixels)

        #: :class:`Tuple <typing.Tuple>` [ :class:`float` , :class:`float` , ... ]: the centroid of the roi
        self.centroid = calculate_centroid(self.xy_vert)[::-1]  # requires vertices!!!

        #: :class:`float`: the radius of the ROI
        self.radius = calculate_radius(self.centroid, self.rc_vert, method="mean")  # requires vertices + centroid!!!

    def __str__(self):
        return f"ROI centered at {tuple([round(val) for val in self.centroid])}"

    @cached_property
    def mask(self) -> NDArray[bool]:
        """
        :Getter: Boolean mask with the dimensions of the reference image indicating the pixels of the ROI
        :Getter Type: :class:`ndarray <numpy.ndarray>` [ :class:`Any <typing.Any>` ,
            :class:`dtype <numpy.dtype>` [ :class:`int` ]]
        :Setter: This property cannot be set
        """
        mask = np.zeros(self.reference_shape, dtype=np.bool_)
        for pair in range(self.rc.shape[0]):
            y, x = self.rc[pair, :]
            mask[y, x] = True
        return mask

    @cached_property
    def xy(self) -> NDArray[int]:
        """
        :Getter: Nx2 array containing the x,y coordinate pairs for the roi
        :Getter Type: :class:`ndarray <numpy.ndarray>` [ :class:`Any <typing.Any>` ,
            :class:`dtype <numpy.dtype>` [ :class:`int` ]]
        :Setter: This property cannot be set
        """
        return np.vstack([self.x_pixels, self.y_pixels]).T

    @cached_property
    def xy_vert(self) -> NDArray[int]:
        """
        :Getter: Nx2 array containing the x,y coordinate pairs comprising the roi's approximate convex hull. Can be
            considered *approximately* the outline of the ROI considering the assumption that the ROI has no
            concavities.
        :Getter Type: :class:`ndarray <numpy.ndarray>` [ :class:`Any <typing.Any>` ,
            :class:`dtype <numpy.dtype>` [ :class:`int` ]]
        :Setter: This property cannot be set
        """
        return self.xy[self.vertices, :]

    @cached_property
    def rc(self) -> NDArray[int]:
        """
        :Getter: Nx2 array containing the r,c coordinate pairs for the roi
        :Getter Type: :class:`ndarray <numpy.ndarray>` [ :class:`Any <typing.Any>` ,
            :class:`dtype <numpy.dtype>` [ :class:`int` ]]
        :Setter: This property cannot be set
        """
        return np.vstack([self.y_pixels, self.x_pixels]).T

    @cached_property
    def rc_vert(self) -> NDArray[int]:
        """
        :Getter: Nx2 array containing the r,c coordinate pairs comprising the roi's approximate convex hull. Can be
            considered *approximately* the outline of the ROI considering the assumption that the ROI has no
            concavities.
        :Getter Type: :class:`ndarray <numpy.ndarray>` [ :class:`Any <typing.Any>` ,
            :class:`dtype <numpy.dtype>` [ :class:`int` ]]
        :Setter: This property cannot be set
        """
        return self.rc[self.vertices, :]

    @property
    def x_pixels(self) -> NDArray[int]:
        """
        :Getter: X-pixels of the roi
        :Getter Type: :class:`ndarray <numpy.ndarray>` [ :class:`Any <typing.Any>` , :class:`dtype <numpy.dtype>`
            [ :class:`int` ]]
        :Setter: The x-pixels are protected from being changed after instantiation
        :Setter Type: :class:`ndarray <numpy.ndarray>` [ :class:`Any <typing.Any>` , :class:`dtype <numpy.dtype>`
            [ :class:`int` ]]
        """
        return self._x_pixels

    @x_pixels.setter
    def x_pixels(self, value: Union[NDArray[int], Sequence[int]]):
        if self._x_set:
            raise PermissionError("Changing the pixel-coordinates after instantiation is not permitted!")
        else:
            self._x_pixels = value
            self._x_set = True

    @property
    def y_pixels(self) -> NDArray[int]:
        """
        :Getter: Y-pixels of the roi
        :Getter Type: :class:`ndarray <numpy.ndarray>` [ :class:`Any <typing.Any>` , :class:`dtype <numpy.dtype>`
            [ :class:`int` ]]
        :Setter: The y-pixels are protected from being changed after instantiation
        :Setter Type: :class:`ndarray <numpy.ndarray>` [ :class:`Any <typing.Any>` , :class:`dtype <numpy.dtype>`
            [ :class:`int` ]]
        """
        return self._y_pixels

    @y_pixels.setter
    def y_pixels(self, value: Union[NDArray[int], Sequence[int]]):
        if self._y_set:
            raise PermissionError("Changing the pixel-coordinates after instantiation is not permitted!")
        else:
            self._y_pixels = value
            self._y_set = True

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

    :param pixels: Nx2 array of x and y-pixel pairs **strictly** in rc form. If this argument is one-dimensional,
        it will be considered as an ordered sequence of x-pixels. The matching y-pixels must be then be provided
        as an additional argument.

    :param ypixels: The y-pixels of the roi if and only if the first argument is one-dimensional.

    :param reference_shape: the shape of the reference image from which the roi was generated

    :param method: the method utilized for generating an approximation of the roi
        ("mean", "bound", "unbound", "ellipse")

    :param plane: index of the imaging plane (if multi-plane)

    :param properties: optional properties to include

    :param zpix: The z-pixels of the roi (if volumetric)
    """
    def __init__(self,
                 pixels: Union[NDArray[int], Sequence[int]],
                 y_pixels: Optional[Union[NDArray[int], Sequence[int]]] = None,
                 reference_shape: Tuple[float, float] = (512, 512),
                 method: str = "literal",
                 plane: Optional[int] = None,
                 properties: Optional[Mapping] = None,
                 z_pixels: Optional[Union[NDArray[int], Sequence[int]]] = None,
                 **kwargs):

        # initialize new attr
        self._method = None

        #: :class:`ApproximateROI`: An approximation of the roi
        self.approximation = None

        # initialize parent attr
        super().__init__(pixels, y_pixels, reference_shape, plane, properties, z_pixels, **kwargs)

        # set method
        self.approx_method = method

    @property
    def approx_method(self) -> str:
        """
        :Getter: Method used for approximating the roi ("literal", "bound", "unbound", "ellipse")
        :Getter Type: :class:`str`
        :Setter: Method used for approximating the roi ("literal", "bound", "unbound", "ellipse")
        :Setter Type: :class:`str` , default: "literal"
        """
        return self._method

    @staticmethod
    def __name__() -> str:
        return "ROI"

    @approx_method.setter
    def approx_method(self, method: str = "literal") -> ROI:
        if method != self._method:
            self.approximation = ApproximateROI(self, method)
            self._method = method


class ApproximateROI(_ROIBase):
    """
    An approximation of an ROI. The approximated ROI is formed by generating an ellipse at the specified centroid with
    a radius calculated by the **method** parameter. Like :class:`ROI <CalSciPy.roi_tools.ROI>`, contains the base
    characteristics & properties of an ROI. Like :class:`ROI <CalSciPy.roi_tools.ROI>`, the properties of this class
    are only calculated once.

    """
    def __init__(self,
                 roi: ROI,
                 method: str = "literal"):
        """
        An approximation of an ROI. The approximated ROI is formed by generating an ellipse at the specified centroid with
        a radius calculated by the **method** parameter. Like :class:`ROI <CalSciPy.roi_tools.ROI>`, contains the base
        characteristics & properties of an ROI. Like :class:`ROI <CalSciPy.roi_tools.ROI>`, the properties of this class
        are only calculated once.

        """

        # cover unimplemented
        if method == "ellipse":
            raise NotImplementedError

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
        """
        :Getter: Method used for calculating radius ("bound", "unbound", "ellipse", "literal")
        :Getter Type: :class:`str`
        :Setter: This property cannot be set
        """
        return self._method

    @staticmethod
    def __name__() -> str:
        return "ROI Approximation"

    @classmethod
    def __pre_init__(cls: ApproximateROI,
                     roi: ROI,
                     method: str = "literal"
                     ) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
        """
        Pre-initialization magic to generate approximated pixels from input arguments

        """
        if method == "literal":
            x_pixels = roi.x_pixels
            y_pixels = roi.y_pixels
        else:
            radius = calculate_radius(roi.centroid, roi.rc_vert, method=method)
            y_pixels, x_pixels = calculate_mask(roi.centroid, radius, roi.reference_shape)
        return x_pixels, y_pixels, roi.reference_shape

    def __repr__(self):
        return "ROI Approximation(" + "".join([f"{key}: {value}, " for key, value in vars(self).items()]) + ")"


class ROIHandler(metaclass=ABCMeta):
    """
    Abstract object for generating reference images and ROI objects

    """

    @staticmethod
    @abstractmethod
    def convert_one_roi(roi: Any, reference_shape: Sequence[int, int] = (512, 512)) -> ROI:
        """
        Abstract method for converting one roi in an ROI object

        :param roi: Some sort of roi data structure

        :type roi: :class:`Any <typing.Any>`

        :param reference_shape: The reference_shape of the image containing the rois

        :type reference_shape: :class:`Sequence <typing.Sequence>` [ :class:`int` , :class:`int` ], default: (512, 512)

        :returns: One ROI object

        :rtype: :class:`ROI <CalSciPy.roi_tools.ROI>`
        """
        ...

    @staticmethod
    @abstractmethod
    def from_file(*args, **kwargs) -> Tuple[Any, Any]:
        """
        Abstract method to load the rois_data_structure and reference_image_data_structure from a file or folder

        :returns: Data structures containing the rois and reference image
        """
        ...

    @staticmethod
    @abstractmethod
    def generate_reference_image(data_structure: Any) -> np.ndarray:
        """
        Abstract method to generate a reference image from some data structure

        :param data_structure: Some data structure from which the reference image can be derived

        :returns: Reference image
        """
        ...

    @classmethod
    def import_rois(cls: ROIHandler,
                    rois: Union[Iterable, np.ndarray],
                    reference_shape: Sequence[int, int] = (512, 512)
                    ) -> dict:
        """
        Abstract method for importing rois

        :param rois: Some sort of data structure iterating over all rois or a numpy array which will be converted to
            an Iterable

        :type rois: :class:`Union <typing.Union>` [ :class:`Iterable <typing.Iterable>` ,
            :class:`ndarray <numpy.ndarray>` ]

        :param reference_shape: The reference_shape of the image containing the rois

        :type reference_shape: :class:`Sequence <typing.Sequence>` [ :class:`int` , :class:`int` ] = (512, 512)

        :returns: Dictionary in which the keys are integers indexing the roi and each roi is an ROI object

        :rtype: :class:`dict`
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


def calculate_radius(centroid: Sequence[Number, Number],
                     vertices_coordinates: NDArray[int],
                     method: str = "mean"
                     ) -> Union[float, Tuple[Tuple[float, float], float]]:
    """
    Calculates the radius of the roi using one of the following methods:

        1. **mean**
            A symmetrical radius calculated as the average distance between the centroid and the vertices of
            the approximate convex hull
        2. **bound**
            A symmetrical radius calculated as the minimum distance between the centroid and the vertices of
            the approximate convex hull - 1
        3. **unbound**
            A symmetrical radius calculated as 1 + the maximal distance between the centroid and the
            vertices of the approximate convex hull
        4. **ellipse**
            An asymmetrical set of radii whose major-axis radius forms the angle theta with respect to the
            y-axis of the reference image

    :param centroid: Centroid of the roi in row-column format (y, x)

    :type centroid: :class:`Sequence <typing.Sequence>` [ :class:`Number <numbers.Number>` ,
        :class:`Number <numbers.Number>` ]

    :param vertices_coordinates: Nx2 array containing the pixels that form the vertices of the approximate convex hull

    :type vertices_coordinates: :class:`ndarray <numpy.ndarray>` [ :class:`Any <typing.Any>` ,
        :class:`dtype <numpy.dtype>` [ :class:`int` ]]

    :param method: Method to use when calculating radius ("mean", "bound", "unbound", "ellipse")

    :type method: :class:`str` , default: 'mean'

    :returns: The radius of the roi for symmetrical radi;
        the major & minor radii and the value of theta for asymmetrical radii

    :rtype: :class:`Union <typing.Union>` [ :class:`float` , :class:`Tuple <typing.Tuple>
        [ :class:`Tuple <typing.Tuple>` [ :class:`Tuple <typing.Tuple>`
        [ :class:`float` , :class:`float` ], :class:`float` ]]
    """

    center = np.asarray(centroid)
    center = np.reshape(center, (1, 2))
    radii = cdist(center, vertices_coordinates)

    if method == "mean":
        return np.mean(radii)
    elif method == "bound":
        return np.min(radii) - 1
    elif method == "unbound":
        return np.max(radii) + 1
    # secret for debugs
    # elif method == "all":
    #    return radii
    else:
        raise NotImplementedError(f"Request method {method} is not supported")


def calculate_centroid(pixels: Union[NDArray[int], Sequence[int]],
                       y_pixels: Optional[Union[NDArray[int], Sequence[int, ...]]] = None
                       ) -> Tuple[float, float]:
    """
    Calculates the centroid of a polygonal roi .The vertices of the roi's approximate convex hull are calculated
    (if necessary) and the centroid estimated from these vertices using the shoelace formula.

    :param pixels: Nx2 array of x and y-pixel pairs in xy or rc form. If this argument is one-dimensional,
        it will be considered as an ordered sequence of x-pixels. The matching y-pixels must be then be provided
        as an additional argument.

    :type pixels: :class:`Union <typing.Union>` [ :class:`ndarray <numpy.ndarray>` [ :class:`Any <typing.Any>` ,
        :class:`dtype <numpy.dtype>` [ :class:`int` ], :class:`Sequence <typing.Sequence>` [ :class:`int` ]]

    :param y_pixels: The y-pixels of the roi if and only if the first argument is one-dimensional.

    :type y_pixels: :class:`Optional <typing.Optional>` [ :class:`Union <typing.Union>`
        [ :class:`ndarray <numpy.ndarray>` [ :class:`Any <typing.Any>` , :class:`dtype <numpy.dtype>` [ :class:`int`
        ], :class:`Sequence <typing.Sequence>` [ :class:`int` ]]], default: None

    :returns: The centroid of the roi. Whether the centroid is in xy or rc form is dependent on the
        form of the arguments

    :rtype: :class:`Tuple <typing.Tuple>` [ :class:`float` , :class:`float` ]
    """
    # if ypix is provided and xpix is one dimensional
    # we have to do it weird this way because many users will pass a 2D array that is of shape (N, 1) rather than (N, )
    y_pixels, pixels = _validate_pixels(pixels, y_pixels)

    # we need RC
    pixels = np.vstack([y_pixels, pixels]).T

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

    :param centroid: Centroid of the roi in row-column form (y, x)

    :type centroid: :class:`Sequence <typing.Sequence>` [ :class:`Number <numbers.Number>` ,
    :class:`Number <numbers.Number>` ]]

    :param radii: Radius of the roi. Only one radius is required if the roi is symmetrical (i.e., circular).
        For an elliptical roi both a long and short radius can be provided.

    :type radii: :class:`Union <typing.Union>` [ :class:`Number <numbers.Number>` ,
        :class:`Sequence <typing.Sequence>` [ :class:`Number <numbers.Number>` , :class:`Number <numbers.Number>` ]]

    :param reference_shape: Dimensions of the reference image the roi lies within. If only one value is provided
        it is considered symmetrical.

    :type reference_shape: :class:`Union <typing.Union>` [ :class:`Number <numbers.Number>` ,
        :class:`Sequence <typing.Sequence>` [ :class:`Number <numbers.Number>` , :class:`Number <numbers.Number>` ]],
        default: None

    :param theta: Angle of the long-radius with respect to the y-axis of the reference image

    :type theta: :class:`Number <numbers.Number>` , default: None

    :returns: Boolean mask identifying which pixels contain the roi within the reference image

    :rtype: :class:`ndarray <numpy.ndarray>` [ :class:`Any <typing.Any>` ,
        :class:`dtype <numpy.dtype>` [ :class:`bool` ]]
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
    if reference_shape is not None:
        bounding_rect[:, 0] = bounding_rect[:, 0].clip(0, reference_shape[0] - 1)
        bounding_rect[:, 1] = bounding_rect[:, 1].clip(0, reference_shape[-1] - 1)

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
    if reference_shape is not None:
        yy.clip(0, reference_shape[0] - 1)
        xx.clip(0, reference_shape[-1] - 1)

    return yy, xx


def identify_vertices(pixels: Union[NDArray[int], Sequence[int]],
                      y_pixels: Optional[Union[NDArray[int], Sequence[int]]] = None
                      ) -> Tuple[int, ...]:
    """
    Identifies the points of a given polygon which form the vertices of the approximate convex hull. This function
    wraps :class:`scipy.spatial.ConvexHull`, which is an ultimately a wrapper for `QHull <https://www.qhull.org>`_.
    It's a fast and easy alternative to actually determining the *true* boundary vertices given the assumption that
    cellular ROIs are convex (i.e., cellular rois ought to be roughly elliptical).

    :param pixels: Nx2 array of x and y-pixel pairs in xy or rc form. If this argument is one-dimensional,
        it will be considered as an ordered sequence of x-pixels. The matching y-pixels must be then be provided
        as an additional argument.

    :type pixels: :class:`Union <typing.Union>` [ :class:`ndarray <numpy.ndarray>` [ :class:`Any <typing.Any>` ,
        :class:`dtype <numpy.dtype>` [ :class:`int` ], :class:`Sequence <typing.Sequence>` [ :class:`int` ]]

    :param y_pixels: The y-pixels of the roi if and only if the first argument is one-dimensional.

    :type y_pixels: :class:`Optional <typing.Optional>` [ :class:`Union <typing.Union>`
        [ :class:`ndarray <numpy.ndarray>` [ :class:`Any <typing.Any>` , :class:`dtype <numpy.dtype>` [ :class:`int`
        ], :class:`Sequence <typing.Sequence>` [ :class:`int` ]]], default: None

    :returns: Index of which points form the vertices of the approximate convex hull.
        It may alternatively be considered an index of the smallest set of pixels that are able to demarcate the
        boundaries of the roi, though this only holds if the polygon doesn't have any concave portions.
    """

    # if not numpy array, convert
    pixels = np.asarray(pixels)

    y_pixels, pixels = _validate_pixels(pixels, y_pixels)
    pixels = np.vstack([y_pixels, pixels]).T

    # approximate convex hull
    hull = ConvexHull(pixels)

    # return the vertices
    return hull.vertices


def _validate_pixels(pixels: Union[NDArray[int], Sequence[int]],
                     y_pixels: Optional[Union[NDArray[int], Sequence[int]]]
                     ) -> Tuple[NDArray[int], NDArray[int]]:

    pixels = np.asarray(pixels)
    if y_pixels is None:
        assert (sum(pixels.shape) >= max(pixels.shape) + 1)
        return pixels[:, 0], pixels[:, 1]
    else:
        y_pixels = np.asarray(y_pixels)
        assert (y_pixels.shape == pixels.shape)
        return y_pixels, pixels
