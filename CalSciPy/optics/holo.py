from __future__ import annotations
from typing import Optional, Any
from importlib import import_module

from slmsuite.hardware.slms.slm import SLM as _SLM


class SLM:
    """
    Wrapper class for `slmsuite <https://github.com/QPG-MIT>`_\'s :class:`SLM class <slmsuite.hardware.slms.slm>`

    """
    def __init__(self,
                 brand: Union[_SLM, str] = _SLM,
                 pixel_pitch: Tuple[Number, Number] = (15, 15),
                 settle_time: Number = 300,
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
        :param kwargs: any kwargs passed to this specific brand of SLM
        """

        width, height = shape

        pixel_pitch_x, pixel_pitch_y = pixel_pitch

        # convert wavelength from nm to um
        wavelength = self._format_wavelength(wavelength)

        # convert settle time from ms to s
        settle_time = self._format_settle_time(settle_time)

        # if brand is a str, retrieve the appropriate class
        if isinstance(brand, str):
            brand = self._detect_slm_brand(brand)

        # make sure the component class is a subclass of slmsuite's abstract SLM class
        assert(issubclass(brand, _SLM))

        #: _SLM: Component class is an instance of slmsuite's abstract SLM class or one of its subclasses
        self._slm = brand(width=width,
                          height=height,
                          wav_um=wavelength,
                          dx_um=pixel_pitch_x,
                          dy_um=pixel_pitch_y,
                          settle_time_s=settle_time,
                          **kwargs)

    @staticmethod
    def _detect_slm_brand(brand: str) -> _SLM:
        # make brand lowercase to find module
        brand = brand.lower()

        # import module
        # _module = import_module("slmsuite.hardware.slms." + brand)
        # TODO: Here until slmsuite fix for PCIe SLMs
        _module = import_module("CalSciPy.optics." + brand)

        # retrieve implementation
        return vars(_module).get(brand.capitalize())

    @staticmethod
    def _format_wavelength(wavelength_in_nm):
        wavelength_in_um = wavelength_in_nm / 1000
        return wavelength_in_um

    @staticmethod
    def _format_settle_time(settle_time_in_ms):
        settle_time_in_s = settle_time_in_ms / 1000
        return settle_time_in_s
