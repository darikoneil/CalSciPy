from __future__ import annotations
from numbers import Number

import numpy as np


"""
Functions useful for designing optical systems
"""


def diff_lim_spot(wavelength: Number,
                  f: Number,
                  beam_diameter: Optional[Number] = None,
                  w0: Optional[Number] = None
                  ) -> float:
    """
    Calculate the size of a diffraction-limited spot.

    :param wavelength: wavelength of the laser source (nm)

    :param f: focal length of the focusing lens (mm)

    :param beam_diameter: beam diameter at 1/e^2 (mm)

    :param w0: beam waist (mm)

    :returns: diameter of the diffraction-limited spot (um)
    """
    # calculate w0 if necessary
    if beam_diameter is not None:
        w0 = beam_diameter / 2

    # convert wavelength from nm to mm
    wavelength = wavelength / (1000**2)

    # calculate spot diameter
    d = (wavelength * f) / w0

    # convert spot diameter to um (selected because it's the most likely unit)
    d *= 1000

    return d


def rayleigh_length(wavelength: Number,
                    beam_diameter: Optional[Number] = None,
                    n: Number = 1,
                    w0: Optional[Number] = None
                    ) -> float:
    """
    Calculate the rayleigh length

    :param wavelength: wavelength of the laser source (nm)

    :param beam_diameter: beam diameter at 1/e^2 (mm)

    :param n: refractive index of medium

    :param w0: beam waist (mm)

    :returns: rayleigh length (mm)
    """
    # calculate w0 if necessary
    if beam_diameter is not None:
        w0 = beam_diameter / 2

    # convert wavelength from nm to mm
    wavelength = wavelength / (1000**2)

    # calculate rayleigh length
    zr = np.pi * (w0 ** 2) * n
    zr /= wavelength

    return zr
