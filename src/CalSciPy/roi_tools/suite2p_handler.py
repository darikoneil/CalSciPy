from __future__ import annotations
from typing import Sequence, Any, Union
from pathlib import Path

import numpy as np
from PPVD.parsing import parameterize
from PPVD.style import TerminalStyle

from .roi_tools import ROI, ROIHandler


class Suite2PHandler(ROIHandler):
    @staticmethod
    def convert_one_roi(roi: Any, reference_shape: Sequence[int, int] = (512, 512)) -> ROI:
        """
        Generates :class:`ROI <CalSciPy.roi_tools.ROI>` from `suite2p <https://www.suite2p.org>`_
        stat array

        :param roi: dictionary containing one suite2p roi

        :type roi: :class:`Any <typing.Any>`

        :param reference_shape: Reference_shape of the reference image containing the roi

        :type reference_shape: :class:`Sequence <typing.Sequence>`\[:class:`int`\, :class:`int`\],
            default: ``(512, 512)``

        :returns: ROI instance for the roi

        :rtype: :class:`ROI <CalSciPy.roi_tools.ROI>`
        """
        xpix = roi.get("xpix")[~roi.get("overlap")]
        ypix = roi.get("ypix")[~roi.get("overlap")]
        return ROI(pixels=xpix,
                   y_pixels=ypix,
                   reference_shape=reference_shape,
                   properties=roi
                   )

    @staticmethod
    @_convert_permitted_types_to_required(permitted=(str, Path), required=Path, pos=0, key="folder")
    def from_file(folder: Union[str, Path], *args, **kwargs) -> Sequence[np.ndarray, dict]:  # noqa: U100
        """
        Loads stat and ops from file

        :param folder: Folder containing `suite2p <https://www.suite2p.org>`_ data. The folder must contain the
            associated *stat.npy* & *ops.npy* files, though it is recommended the folder also contain the *iscell.npy*
            file.

        :type folder: :class:`Union <typing.Union>`\[:class:`str`\, :class:`Path <pathlib.Path>`\]

        :returns: Stat and ops

        :rtype: :class:`Sequence <typing.Sequence>`\[:class:`ndarray <numpy.ndarray>`\, :class:`dict`\]
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
         Generates an appropriate reference image from `suite2p <https://www.suite2p.org>`_ ops dictionary

        :param data_structure: Ops dictionary

        :returns: Reference image
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


@parameterize
def _convert_permitted_types_to_required(function: Callable,
                                          permitted: Tuple,
                                          required: Any,
                                          pos: int = 0,
                                          key: str = "folder") -> Callable:
    """
    Decorator that converts a tuple of permitted types to type supported by the decorated method

    :param function: function to be decorated
    :type function: Callable
    :param permitted: permitted types
    :type permitted: tuple
    :param required: type required by code
    :type required: Any
    :param pos: index of argument to be converted
    :type pos: int
    """
    @wraps(function)
    def decorator(*args, **kwargs):
        if key in kwargs:
            use_args = False
            allowed_input = kwargs.get(key)
        elif args is not None and pos is not None:
            use_args = True
            allowed_input = args[pos]
        else:
            return function(*args, **kwargs)
        if isinstance(allowed_input, permitted):
            allowed_input = required(allowed_input)
        if not isinstance(allowed_input, required):
            raise TypeError(f"{TerminalStyle.GREEN}Input {pos}: {TerminalStyle.YELLOW}"
                            f"inputs are permitted to be of the following types "
                            f"{TerminalStyle.BLUE}{permitted} {TerminalStyle.RESET}")
        if use_args:
            args = amend_args(args, allowed_input, pos)
        else:
            kwargs[key] = allowed_input
        # noinspection PyArgumentList
        return function(*args, **kwargs)
    return decorator
