from __future__ import annotations
from typing import Tuple
from numbers import Number

import numpy as np
from numpy.typing import NDArray
from slmsuite.holography.algorithms import Hologram as _Hologram
from slmsuite.hardware.slms.slm import SLM as _SLM

from ..roi_tools import ROI
from .opto_objects import StimulationSequence, StimulationGroup

"""
Functions related to computer-generated holography. Generally this module wraps
neuroscience-specific functionality onto SLMSuite.
"""

"""
Jargon notes:
    * near-field -> the SLM surface
    * far-field -> where you are actually imaging/stimulating
"""


def generate_target_mask(group: StimulationGroup, method: str = "radius") -> NDArray[bool]:

    #rois = [getattr(roi.mask, method) for roi in group.rois]
    rois = [roi for roi in group.rois]

    shape = group.shape

    masks = [roi.mask for roi in rois]

    #for roi in rois:
        #one_mask = np.zeros(shape, dtype=bool)
        #for pt in range(coordinates.shape[0]):
            #one_mask[coordinates[pt, 0], coordinates[pt, 1]] = True
        #masks.append()

    if group.point_interval == 0:
        masks = np.sum(masks, axis=0)

    return masks


class Hologram(_Hologram):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class SLM(_SLM):
    """
    Wrapper class for SLMSuite's SLM class
    """
    def __init__(self,
                 brand: str,
                 pixel_pitch: Tuple[Number, Number] = (10, 10),
                 settle_time: Number = 0.3,
                 shape: Tuple[int, int] = (512, 512),
                 wavelength: Number = 1040,
                 **kwargs
                 ):
        """
        Initialize wrapper class

        :param brand: brand of slm
        :param pixel_pitch: x, y
        :param settle_time: settle time in ms
        :param shape: x pixels, y pixels
        :param wavelength: wavelength in nm
        :param kwargs: any kwargs passed to superclass
        """
        width, height = shape

        pixel_pitch_x, pixel_pitch_y = pixel_pitch

        wavelength = self._format_wavelength(wavelength)

        settle_time = self._format_settle_time(settle_time)

        super().__init__(width=width,
                         height=height,
                         wav_um=wavelength,
                         dx_um=pixel_pitch_x,
                         dy_um=pixel_pitch_y,
                         settle_time_s=settle_time,
                         **kwargs)

    @staticmethod
    def _format_wavelength(wavelength_in_nm):
        wavelength_in_um = wavelength_in_nm / 1000
        return wavelength_in_um

    @staticmethod
    def _format_settle_time(settle_time_in_ms):
        settle_time_in_s = settle_time_in_ms / 1000
        return settle_time_in_s
