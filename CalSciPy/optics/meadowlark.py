"""
Hardware control for Meadowlark SLMs.
Tested with Meadowlark (AVR Optics) P1920-400-800-HDMI-T.

Note
~~~~
Check that the Blink SDK, including DLL files etc., are in the default folder
or otherwise pass the correct directory in the constructor.
"""
import os
import ctypes
import warnings

from slmsuite.hardware.slms.slm import SLM


# DEFAULT_SDK_PATH = "C:\\Program Files\\Meadowlark Optics\\Blink 1920 HDMI\\"
DEFAULT_SDK_PATH = "C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus\\"


class Meadowlark(SLM):
    """
    Interfaces with Meadowlark SLMs.

    Attributes
    ----------
    slm_lib : ctypes.CDLL
        Connection to the Meadowlark library.
    sdk_path : str
        Path of the Blink SDK folder.
    """

    def __init__(self, verbose=True, sdk_path=DEFAULT_SDK_PATH, lut_path=None, dx_um=8, dy_um=8, hdmi=True, **kwargs):
        r"""
        Initializes an instance of a Meadowlark SLM.

        Caution
        ~~~~~~~
        :class:`.Meadowlark` defaults to 8 micron SLM pixel size
        (:attr:`.SLM.dx_um` = :attr:`.SLM.dy_um` = 8).
        This is valid for most Meadowlark models, but not true for all!

        Arguments
        ---------
        verbose : bool
            Whether to print extra information.
        sdk_path : str
            Path of the Blink SDK folder. Stored in :attr:`sdk_path`.
        lut_path : str OR None
            Passed to :meth:`load_lut`.
        kwargs
            See :meth:`.SLM.__init__` for permissible options.
        """

        # Added flag here for checking whether SDK is HDMI or PCIe
        self.hdmi = hdmi

        # Validates the DPI awareness of this context, which is presumably important for scaling.
        if verbose: print("Validating DPI awareness...", end="")

        awareness = ctypes.c_int()
        error_get = ctypes.windll.shcore.GetProcessDpiAwareness(0, ctypes.byref(awareness))
        error_set = ctypes.windll.shcore.SetProcessDpiAwareness(2)
        success = ctypes.windll.user32.SetProcessDPIAware()

        if not success:
            raise RuntimeError(
                "Meadowlark failed to validate DPI awareness."
                "Errors: get={}, set={}, awareness={}".format(error_get, error_set, awareness.value)
            )
        if verbose: print("success")

        # Open the SLM library
        if verbose: print("Constructing Blink SDK...", end="")

        dll_path = os.path.join(sdk_path, "SDK", "Blink_C_wrapper")
        try:
            ctypes.cdll.LoadLibrary(dll_path)
            self.slm_lib = ctypes.CDLL("Blink_C_wrapper")
        except:
            print("failure")
            raise ImportError(
                "Meadowlark .dlls did not did not import correctly. Is '{}' the correct path?"
                .format(dll_path)
            )

        self.sdk_path = sdk_path

        # Initialize is different between HDMI and PCIe SDKS. Added logic here that is specific to each SDK
        if self.hdmi:
            # Initialize the SDK. The requirements of Matlab, LabVIEW, C++ and Python are different, so pass
            # the constructor a boolean indicating if we are calling from C++/Python (true), or Matlab/LabVIEW (false)
            bool_cpp_or_python = ctypes.c_uint(1)
            self.slm_lib.Create_SDK(bool_cpp_or_python)

            # Adjust pre- and post-ramp slopes for accurate voltage setting
            # (otherwise, custom LUT calibration is not properly implemented [this feature is not implemented in
            # slmsuite]).You may need a special version of the SDK sent to you from Meadowlark to have access to
            # these parameters.
            # self.slm_lib.SetPreRampSlope(20) # default is 7
            # self.slm_lib.SetPostRampSlope(24) # default is 24

            slm_width = self.slm_lib.Get_Width()
            slm_height = self.slm_lib.Get_Height()
            slm_depth = self.slm_lib.Get_Depth()
        else:
            # Initialize the SDK. Requires some additional parameters than HDMI. Here I am just checking for them in
            # kwargs to avoid clogging up rest of code, but in future if PCIe is a separate class then it might be
            # useful for some users to have them as documented arguments
            sdk_keys = {
                "bit_depth": ctypes.c_uint(8),
                "is_nematic_type": ctypes.c_bool(1),
                "RAM_write_enable": ctypes.c_bool(1),
                "use_GPU": ctypes.c_bool(1),
                "max_transients": ctypes.c_uint(20),
                "true_frames": ctypes.c_int(3)
            }

            for key in kwargs.keys():
                if key in sdk_keys:
                    sdk_keys[key] = kwargs.get(key)
                    kwargs.pop(key)

            # you could use this for "info" but that method is static so would require some changes
            num_boards_found = ctypes.c_uint(0)

            constructed_okay = ctypes.c_bool(0)

            self.slm_lib.Create_SDK(sdk_keys.get("bit_depth"),
                                    ctypes.byref(num_boards_found),
                                    ctypes.byref(constructed_okay),
                                    sdk_keys.get("is_nematic_type"),
                                    sdk_keys.get("RAM_write_enable"),
                                    sdk_keys.get("use_GPU"),
                                    sdk_keys.get("max_transients")
                                    )

            assert (constructed_okay.value == 1), "Meadowlark SDK was not constructed successfully"

            # assuming board = 1 since multi-boards not supported with rest of class
            slm_width = self.slm_lib.Get_image_width(1)
            slm_height = self.slm_lib.Get_image_height(1)
            slm_depth = self.slm_lib.Get_image_depth(1)

            self.slm_lib.Set_true_frames(sdk_keys.get("true_frames"))

        if verbose: print("success")

        # Load LUT.
        if verbose: print("Loading LUT file...", end="")

        try:
            true_lut_path = self.load_lut(lut_path)
        except RuntimeError:
            print("failure\n(could not find .lut file)")
        else:
            if verbose and true_lut_path != lut_path:
                print("success\n(loaded from '{}')".format(true_lut_path))

        # Construct other variables.
        super().__init__(
            slm_width,
            slm_height,
            bitdepth=slm_depth,
            name="Meadowlark",
            dx_um=dx_um,
            dy_um=dy_um,
            **kwargs
        )

        if self.bitdepth > 8:
            warnings.warn(
                "Bitdepth of {} > 8 detected; this has not been tested and might fail.".format(self.bitdepth)
            )

        self.write(None)

    def load_lut(self, lut_path=None):
        """
        Loads a voltage lookup table (LUT) to the SLM.
        This converts requested phase values to physical voltage perturbing
        the liquid crystals.

        Parameters
        ----------
        lut_path : str OR None
            Path to look for an LUT file in.
            If this is a .lut file, then this file is loaded to the SLM.
            If this is a directory, then searches all files inside the
            directory, and loads either the first .lut file, or if possible
            an .lut file starting with `"slm"`
            (which is more likely to correspond to the LUT customized to an SLM,
            as Meadowlark sends such files prefixed by `"slm"` such as `"slm5758_at532.lut"`).

        Raises
        ------
        RuntimeError
            If a .lut file is not found.

        Returns
        -------
        str
            The path which was used to load the LUT.
        """
        # If a path is not given, search inside the SDK path.
        if lut_path is None:
            lut_path = os.path.join(self.sdk_path, "LUT Files")

        # If we already have a .lut file, proceed.
        if len(lut_path) > 4 and lut_path[-4:] == ".lut":
            pass
        else:   # Otherwise, treat the path like a folder and search inside the folder.
            lut_file = None

            for file in os.listdir(lut_path):
                # Only examine .lut files.
                if len(file) >= 4 and file[-4:].lower() == ".lut":
                    # Choose the first one.
                    if lut_file is None:
                        lut_file = file

                    # Or choose the first one that starts with "slm"
                    if file[:3].lower() == "slm" and not lut_file[:3].lower() == "slm":
                        lut_file = file
                        break

            # Throw an error if we didn't find a .lut file.
            if lut_file is not None:
                lut_path = os.path.join(lut_path, lut_file)
            else:
                raise RuntimeError(
                    "Could not find a .lut file at path '{}'".format(lut_path)
                )
        if self.hdmi:
            # Finally, load the lookup table.
            self.slm_lib.Load_lut(lut_path)
        else:
            # Slightly different call here
            self.slm_lib.Load_LUT_file(lut_path)

        return lut_path

    @staticmethod
    def info(verbose=True):
        """
        The normal behavior of this function is to discover the names of all the displays
        to help the user identify the correct display. However, Meadowlark software does
        not currently support multiple SLMs, so this function instead raises an error.

        Parameters
        ----------
        verbose : bool
            Whether to print the discovered information.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError(
            "Meadowlark software does not currently support multiple SLMs, "
            "so a function to identify SLMs is moot. "
            "If functionality with multiple SLMs is desired, contact them directly."
        )

    def close(self):
        """
        See :meth:`.SLM.close`.
        """
        self.slm_lib.Delete_SDK()

    def _write_hw(self, display):
        """
        See :meth:`.SLM._write_hw`.
        """
        if self.hdmi:
            self.slm_lib.Write_image(
                display.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
                ctypes.c_uint(self.bitdepth == 8)   # Is 8-bit
            )
        else:
            self.slm_lib.Write_image(1, display.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
                                     ctypes.c_uint(self.shape[0]*self.shape[1]*self.bitdepth))

    ### Additional Meadowlark-specific functionality

    def get_temperature(self):
        """
        Read the temperature of the SLM.

        Returns
        -------
        float
            Temperature in degrees celcius.
        """
        if self.hdmi:
            return self.slm_lib.Get_SLMTemp()
        else:
            self.slm_lib.Read_SLM_temperature.restype = c_double
            return self.slm_lib.Read_SLM_temperature(1)
            # assuming board = 1 since multi-boards not supported with rest of class

    def get_coverglass_voltage(self):
        """
        Read the voltage of the SLM coverglass.

        Returns
        -------
        float
            Voltage of the SLM coverglass.
        """
        if self.hdmi:
            return self.slm_lib.Get_SLMVCom()
        else:
            self.slm_lib.Get_cover_voltage.restype = c_double
            return self.slm_lib.Get_cover_voltage(1)
            # assuming board = 1 since multi-boards not supported with rest of class
