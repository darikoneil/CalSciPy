from __future__ import annotations
from typing import Tuple, Union, Sequence
from pathlib import Path
from functools import partial, cached_property
from collections import ChainMap
from numbers import Number

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist
from .bruker.xml_objects import GalvoPoint, GalvoPointList


class Photostimulation:
    """
    Photostimulation object that defines patterned photostimulation during an optogenetic experiment

    :ivar rois: dictionary containing a collection of ROI objects for potential photostimulation
    :type rois: dict
    :ivar reference_image: a reference image containing the provided ROIs.
    :type reference_image: numpy.ndarray
    :ivar sequence: the sequence of individual photostimulation events
    """
    def __init__(self, rois: dict, reference_image: np.ndarray = None):
        """
        PPhotostimulation object that defines patterned photostimulation during an optogenetic experiment

        :param rois: dictionary containing a collection of ROI objects for potential photostimulation
        :type rois: dict
        :param reference_image: a reference image containing the provided ROIs.
        :type reference_image: numpy.ndarray
        """
        self.rois = rois
        self.reference_image = reference_image
        self.sequence = None

    @staticmethod
    def _suite2p_roi(idx: int, stat: np.ndarray, shape: Tuple[int, int]) -> ROI:
        """
        Static method generating an ROI for each roi in stat

        :param stat: array containing suite2p stats for each roi
        :param idx: idx of specific roi
        :return: ROI instance for the roi 'idx'
        """
        aspect_ratio = stat[idx].get("aspect_ratio")
        radius = stat[idx].get("radius")
        xpix = stat[idx].get("xpix")[~stat[idx].get("overlap")]
        ypix = stat[idx].get("ypix")[~stat[idx].get("overlap")]
        return ROI(aspect_ratio=aspect_ratio, radius=radius, shape=shape, xpix=xpix, ypix=ypix)

    @classmethod
    def convert_suite2p_rois(cls: Photostimulation, suite2p_rois: np.ndarray, shape: Tuple[int, int] = (512, 512)) -> dict:
        """
        Class method that generates the roi dictionary from provided suite2p stat array

        :param suite2p_rois: array containing suite2p stats for each roi
        :type suite2p_rois: numpy.ndarray
        :param shape: dimensions of image
        :return: dictionary containing a collection of ROI objects for potential photostimulation
        """
        converter = partial(cls._suite2p_roi, stat=suite2p_rois, shape=shape)
        return dict(enumerate([converter(idx) for idx in range(suite2p_rois.shape[0])]))

    @classmethod
    def import_suite2p(cls: Photostimulation, folder: Path, shape: Tuple[int, int] = (512, 512)) -> Photostimulation:
        """
        Class method which builds a photostimulation instance given suite2p data

        :param folder: folder containing suite2p data. The folder must contain the associated "stat.npy" file,
            though it is recommended the folder also contain the "iscell.npy" and "ops.npy" files
        :param shape: if the provided folder does not contain the ops.npy file this value is used generate an
            approximate reference image
        :return: An instance of Photostimulation
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

        # if ops is provided then retrieve the reference image
        try:
            ops = np.load(folder.joinpath("ops.npy"), allow_pickle=True).item()
        except FileNotFoundError:
            reference_image = np.ones(shape, )
        else:
            reference_image = ops.get("Vcorr")

        # convert rois in stat to the expected form
        rois = cls.convert_suite2p_rois(stat, reference_image.shape)

        # generate instance
        return Photostimulation(rois, reference_image)


    def generate_galvo_point(self, idx: int, parameters: dict = None) -> GalvoPoint:
        roi = self.rois[idx]
        y, x = roi.coordinates
        name = f"Point {idx}"
        index = idx
        spiral_size = roi.mask.bound_radius

        roi_properties = {key: value for key, value in zip(["y", "x", "name", "index", "spiral_size"],
                                                           [y, x, name, index, spiral_size])}

        if parameters is not None:
            roi_properties = ChainMap(parameters, roi_properties)

        return GalvoPoint(**roi_properties)

    def generate_galvo_point_list(self, parameters: dict = None) -> GalvoPointList:
        galvo_points = tuple([self.generate_galvo_point(idx, parameters) for idx in self.rois])
        return GalvoPointList(galvo_points=galvo_points)

    @property
    def targets(self) -> int:
        return 15

    def __str__(self):
        return f"Photostimulation experiment targeting {self.targets} neurons from {len(self.rois)} total " \
               f"ROIs within {self.reference_image.shape[0]} x {self.reference_image.shape[1]} reference image (x, y)"

    @staticmethod
    def __name__() -> str:
        return "Photostimulation"


class Group:
    def __init__(self):
        """
        Photostimulation group object containing the index of rois to stimulate
        and relevant stimulation parameters

        :ivar order: a tuple containing the identity and stimulation order of the rois in this group
        :type order: Tuple[int]
        :ivar repetitions: an integer indicating the number of times to repeat the stimulation
        """
        self.order = None
        self.repetitions = 1


class ROI:
    """
     ROI object containing the relevant properties of some specific ROI for using in generating photostimulation
    protocols.

    :param aspect_ratio: ratio of the short-to-long radius of the roi
    :type aspect_ratio: float = 1.0
    :param radius: approximate radius of the ROI
    :type radius: float
    :param shape: the shape of the image from which the roi was generated
    :type shape: Tuple[int, int]
    :param xpix: a 1D numpy array or Sequence indicating the x-pixels of the roi (column-wise)
    :type xpix: numpy.ndarray
    :param ypix: a 1D numpy array or Sequence indicating the y-pixels of the roi (row-wise)
    :type ypix: numpy.ndarray

    :ivar vertices: a 1D tuple indexing the convex hull of the roi (alternatively, may be considered an index of
        the pixels that form the boundaries of the roi
    :type vertices: Tuple[int]
    :ivar coordinates: the centroid of the roi (y, x) calculated using the shoelace approximation
    :type coordinates: Tuple[ float, float]
    :ivar mask: photostimulation mask for the roi
    :type mask: Mask
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
                 aspect_ratio: float = 1.0,
                 radius: float = None,
                 shape: Tuple[float, float] = (512, 512),
                 xpix: np.ndarray = None,
                 ypix: np.ndarray = None):
        """
        ROI object containing the relevant properties of some specific ROI for using in generating photostimulation
        protocols.

        :param aspect_ratio: ratio of the short-to-long radius of the roi
        :type aspect_ratio: float = 1.0
        :param radius: approximate radius of the ROI
        :type radius: float
        :param shape: the shape of the image from which the roi was generated
        :type shape: Tuple[int, int]
        :param xpix: a 1D numpy array or Sequence indicating the x-pixels of the roi (column-wise)
        :type xpix: numpy.ndarray
        :param ypix: a 1D numpy array or Sequence indicating the y-pixels of the roi (row-wise)
        :type ypix: numpy.ndarray
        """
        self.aspect_ratio = aspect_ratio
        self.radius = radius
        self.shape = shape
        self.xpix = xpix
        self.ypix = ypix

        self.vertices = calculate_vertices(self.xpix, self.ypix)
        self.coordinates = calculate_centroid(self.xy_vert)[::-1]  # requires vertices!!!
        self.mask = Mask(self.coordinates, self.rc_vert, self.adj_radii, 0, self.shape)  # requires vertices!!!

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

    def __str__(self):
        return f"ROI centered at {tuple([round(val) for val in self.coordinates])}"

    def __repr__(self):
        return "ROI(" + "".join([f"{key}: {value} " for key, value in vars(self).items()]) + ")"

    @staticmethod
    def __name__() -> str:
        return "ROI"


class Mask:
    """
    Photostimulation Mask object associated with some ROI. Generated by approximating the ROI as an ellipse. The bound
    mask is calculated by constraining the radius of the ellipse to the smaller radius (symmetry constraint)

    :param center: the centroid of the roi (y, x) calculated using the shoelace approximation
    :type center: Tuple[float, float]
    :param rc_vert: Nx2 array containing the r,c coordinate pairs comprising the roi's convex hull approximation
    :type rc_vert: numpy.ndarray
    :param radii: the long and short radii of the roi (long, short)
    :type radii: Tuple[float, float]
    :param theta: angle of the long axis of the roi with respective to the y-axis
    :type theta: float
    :param shape: the shape of the image from which the roi was generated
    :type shape: Tuple[int, int]

    :ivar xy: Nx2 array containing x,y coordinate pairs for the mask
    :type xy: numpy.ndarray
    :ivar xy_vert: Nx2 array containing the x,y coordinate pairs comprising the mask's convex hull approximation
    :type xy_vert: numpy.ndarray
    :ivar rc: Nx2 array containing the r,c coordinate pairs for the mask
    :type rc: numpy.ndarray
    :ivar rc_vert: Nx2 array containing the r,c coordinate pairs comprising the mask's convex hull approximation
    :type rc_vert: numpy.ndarray
    :ivar bound_xy: Nx2 array containing x,y coordinate pairs for the mask
    :type bound_xy: numpy.ndarray
    :ivar bound_xy_vert: Nx2 array containing the x,y coordinate pairs comprising the mask's convex hull approximation
    :type bound_xy_vert: numpy.ndarray
    :ivar bound_rc: Nx2 array containing the r,c coordinate pairs for the mask
    :type bound_rc: numpy.ndarray
    :ivar bound_rc_vert: Nx2 array containing the r,c coordinate pairs comprising the mask's convex hull approximation
    :type bound_rc_vert: numpy.ndarray
    :ivar bound_radius: radius used for constraining the bound mask
    :type bound_radius: float
    """
    def __init__(self,
                 center: Tuple[float, float],
                 rc_vert: np.ndarray,
                 radii: Tuple[float, float],
                 theta: float,
                 shape: Tuple[int, int]):
        """
        Photostimulation Mask object associated with some ROI

        :param center: the centroid of the roi (y, x) calculated using the shoelace approximation
        :type center: Tuple[float, float]
        :param rc_vert: Nx2 array containing the r,c coordinate pairs comprising the roi's convex hull approximation
        :type rc_vert: numpy.ndarray
        :param radii: the long and short radii of the roi (long, short)
        :type radii: Tuple[float, float]
        :param theta: angle of the long axis of the roi with respective to the y-axis
        :type theta: float
        :param shape: the shape of the image from which the roi was generated
        :type shape: Tuple[int, int]

        """

        self.center = center
        self._rc_vert = rc_vert
        self.radii = radii
        self.theta = theta
        self.shape = shape

    @staticmethod
    def __name__() -> str:
        return "Photostimulation Mask"

    def __repr__(self):
        return "Photostimulation Mask(" + "".join([f"{key}: {value} " for key, value in vars(self).items()]) + ")"

    def __str__(self):
        return f"Photostimulation mask centered at {self.center} with radii {self.radii} (bound: {self.bound_radius})" \
               f"with theta {self.theta}'"

    @cached_property
    def _mask(self) -> np.ndarray:
        """
        Photostimulation mask calculated using center, long/short radii, and theta constrained to lie within shape

        """
        y, x = calculate_mask(self.center, self.radii, self.shape)
        return np.vstack([x, y]).T

    @cached_property
    def _mask_vertices(self) -> Tuple[int]:
        """
        Indices of photostimulation points comprising the convex hull approximation of the photostimulation mask

        """
        hull = ConvexHull(self._mask)
        return hull.vertices

    @cached_property
    def bound_radius(self) -> float:
        """
        Radius used for constraining the bound mask

        """
        return calculate_bounding_radius(self.center, self.rc_vert)

    @cached_property
    def _bound_mask(self) -> np.ndarray:
        """
        Bound photostimulation mask calculated using center, the bound radius, and constrained to lie within shape

        """
        y, x = calculate_mask(self.center, self.bound_radius, self.shape)
        return np.vstack([x, y]).T

    @cached_property
    def _bound_vertices(self) -> Tuple[int]:
        """
        Indices of photostimulation points comprising the convex hull approximation of the bound photostimulation mask

        """
        hull = ConvexHull(self._bound_mask)
        return hull.vertices

    @property
    def bound_xy(self) -> np.ndarray:
        """
        Nx2 array containing x,y coordinate pairs for the mask

        """
        return self._bound_mask

    @property
    def bound_rc(self) -> np.ndarray:
        """
        Nx2 array containing the r,c coordinate pairs for the mask

        """
        return np.vstack([self._bound_mask[:, 1], self._bound_mask[:, 0]]).T

    @property
    def bound_xy_vert(self) -> np.ndarray:
        """
        Nx2 array containing the x,y coordinate pairs comprising the mask's convex hull approximation

        """
        return self.bound_xy[self._bound_vertices, :]

    @property
    def bound_rc_vert(self) -> np.ndarray:
        """
        Nx2 array containing the r,c coordinate pairs comprising the mask's convex hull approximation

        """
        return self.bound_rc[self._bound_vertices, :]

    @property
    def xy(self) -> np.ndarray:
        """
        Nx2 array containing x,y coordinate pairs for the mask

        """
        return self._mask

    @property
    def rc(self) -> np.ndarray:
        """
        Nx2 array containing the r,c coordinate pairs for the mask

        """
        return np.vstack([self._mask[:, 1], self._mask[:, 0]]).T

    @property
    def xy_vert(self) -> np.ndarray:
        """
        Nx2 array containing the x,y coordinate pairs comprising the mask's convex hull approximation

        """
        return self.xy[self._mask_vertices, :]

    @property
    def rc_vert(self) -> np.ndarray:
        """
        Nx2 array containing the r,c coordinate pairs comprising the mask's convex hull approximation

        """
        return self.rc[self._mask_vertices, :]


def calculate_centroid(vertices: np.ndarray) -> Tuple[int, int]:
    """
    Calculates the centroid of a polygonal roi given an Nx2 numpy array containing the vertices of the polygon's
    convex hull using the shoelace formula

    :param vertices: vertices of the polygon's convex hull (boundary points)
    :type vertices: numpy.ndarray
    :return: a tuple containing the centroid of the polygon
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


def calculate_mask(center: Sequence[Number, Number],
                   radii: Union[Number, Sequence[Number, Number]],
                   shape: Union[Number, Sequence[Number, Number]] = None,
                   theta: Number = None,
                   ) -> np.ndarray:
    """
    Calculates a photostimulation mask for an elliptical roi at center with radii at theta with respect to the y-axis
    and constrained to lie within shape

    :param center: center of the roi (y, x)
    :type center: Sequence[Number, Number]
    :param radii: radius of the roi. can provide one radius for a symmetrical roi or a long and short radius.
    :type radii: Union[Number, Sequence[Number, Number]]
    :param shape: dimensions of the image the roi lies within. If only one value is provided it is considered
        symmetrical.
    :type shape: Union[Number, Sequence[Number, Number]] = None
    :param theta: angle of the long-radius with respect to the y-axis
    :type theta: Number
    :return: photostimulation mask
    """
    # ensure center is numpy array
    center = np.asarray(center)

    # make sure radii contains both x & y directions
    try:
        assert (len(radii) == 2)
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


def calculate_vertices(xpix: Union[np.ndarray, Sequence[int]], ypix: Union[np.ndarray, Sequence[int]]) -> Tuple[int]:
    """
    Calculate the index of points comprising the convex hull the polygon

    :param xpix: a 1D numpy array or Sequence indicating the x-pixels of the roi (column-wise)
    :param ypix: a 1D numpy array or Sequence indicating the y-pixels of the roi (row-wise)
    :return:  a 1D tuple indexing the convex hull of the roi (alternatively, may be considered an index of
            the pixels that form the boundaries of the roi)
    """
    hull = ConvexHull(np.vstack([xpix, ypix]).T)
    return hull.vertices


# def preview_stimulation_masks(self, ref_image: np.ndarray = None) -> None:
#    if ref_image is None:
#        ref_image = np.zeros((self.image_height, self.image_width))
#
#    with plt.style.context("CalSciPy.main"):
#        # setup
#
#        fig, ax = plt.subplots(1, 1)
#        ax.set_xlim(0, self.image_width)
#        ax.set_ylim(self.image_height, 0)
#        ax.set_title("Photostimulation Targets (ROI, Group)")
#        ax.set_xlabel("X Pixels")
#        ax.set_ylabel("Y Pixels")
#        ax.xaxis.set_major_locator(MultipleLocator(64))
#        ax.yaxis.set_major_locator(MultipleLocator(64))
#        ax.set_axisbelow(True)
#        ax.imshow(ref_image, cmap="Spectral_r")

#        # iterate
#        for roi in self.rois:
#            label_ = f"({roi.index}, {roi.stimulation_group})"
#            y = [y for _, y in roi.vertices]
#            x = [x for x, _ in roi.vertices]
#            vtx_pts = np.vstack([y, x]).T
#            vtx_pts = [vtx_pts[:, 1], vtx_pts[:, 0]]
#            pg = Polygon(vtx_pts, edgecolor=COLORS.red, lw=3, fill=False, label=label_)
#            ax.add_patch(pg)


# def true_mask(mask, rc):
#    mask_coords = {(y, x) for y, x in zip(mask[0].tolist(), mask[1].tolist())}
#    rc_coords = {(y, x) for y, x in zip(rc[:, 0].tolist(), rc[:, 1].tolist())}
#    coords = set.intersection(mask_coords, rc_coords)
#    y = [y for y, _ in list(coords)]
#    x = [x for _, x in list(coords)]
#    return np.vstack([y, x]).T


def calculate_bounding_radius(center: Sequence[Number, Number], vertices: np.ndarray) -> float:
    """
    Calculates the bounding radius of the roi, defined as the shortest distance between the centroid and any point that
        lies within the boundary of the convex hull approximation

    :param center: center of the roi (y, x)
    :param vertices: Nx2 array containing the points that lie on the boundary of the convex hull approximation
    :return: bounding radius
    """
    func_handle = partial(_calculate_bounding_radius, center=center)
    return np.min([func_handle(point=vertices[point, :]) for point in range(vertices.shape[0])])


def _calculate_bounding_radius(center: Sequence[Number, Number], point: np.ndarray) -> float:
    """
    Calculates the distance between the centroid and a specific point

    :param center: the center of the roi (y, x)
    :param point: a specific point (y, x)
    :return: the distance between the center and the point
    """
    coordinate_pair = np.empty((2, 2))
    coordinate_pair[0, :] = center
    coordinate_pair[1, :] = point
    return pdist(coordinate_pair)
