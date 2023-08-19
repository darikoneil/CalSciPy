from __future__ import annotations
from typing import Any
from socket import gethostname, gethostbyname
from getpass import getuser, getpass
from functools import cached_property
from array import array
from enum import Enum

from memoization import cached

#  try:


import win32com  # noqa: F401
import win32com.client as client  # noqa: F401
from win32com.client import CDispatch  # noqa: F401


#  except ImportError:
#    print("Unable to locate pywin32 installation")


"""
Interface for PrairieView Software -- Tested on PrairieView 5.8.0 (Released 3/03/2023) -- EXPERIMENTAL
"""


USER = getuser()


class PrairieLink:
    def __init__(self, system_id: str = "TEST", user_: str = USER, pass_: str = None, auto_run: bool = True):
        """
        Python Interface for Interacting with PrairieView Software through the PrairieLink64 API

        :param system_id: prairieview system id
        :param user_: optional argument to provide username
        :param pass_: optional argument to provide password
        """
        #: str: username
        self._username = user_
        #: str: system id
        self._system_id = system_id
        #: CDispatch: PrairieLink Application
        self._prairie_link = client.Dispatch("PrairieLink64.Application")
        #: str: hostname of the machine currently executing python interpreter
        self._host_name = gethostname()
        #: str: host ip address
        self._ip_address = gethostbyname(self._host_name)
        if auto_run:
            self.connect(password=pass_)

    @property
    def connected(self) -> bool:
        """
        Property indicates whether currently connected to a prairieview instance. This does not reflect whether the
        PrairieLink object is connected to an instance of PrairieLink

        :return: status of connection to prairieview instance
        """
        return self._prairie_link.Connected()

    @property
    def dropped_data(self) -> bool:
        """
        Indicates that data has been dropped from the current acquisition if True/
        Otherwise, all data may be considered successfully written to file.

        :return: indicator of dropped data
        """
        return self._prairie_link.DroppedData()

    @property
    def lines_per_frame(self) -> int:
        """
        Indicates the number of line scanned per frame. This is synonymous with the height / number of y-pixels of
        resulting images.

        :return: lines per frame
        """
        return self._prairie_link.LinesPerFrame()

    @property
    def pixels_per_line(self) -> int:
        """
        Indicates the number of pixels per line. This is synonymous with the width or number of x-pixels.

        :return: pixels per line
        """
        return self._prairie_link.PixelsPerLine()

    @property
    def samples_per_pixel(self) -> int:
        """
        Indicates the number of samples per pixel. This property is necessary for successfully parsing the raw data
        stream since each pixel will be consecutively duplicated a number of times before the next. That is, one cannot
        parse the raw data into the proper shape (frames, y-pixels, x-pixels) without first merging each of these
        samples into a single scalar value

        :return: samples per pixel
        """
        return self._prairie_link.SamplesPerPixel()

    @cached_property
    def version(self) -> str:
        """
        Property indicates the version of connected prairieview software. Uses cached property because it is not
        expected to change within an instance

        :return: version of connected prairieview instance
        """
        return self._prairie_link.Version()

    def buffered_read(self) -> Any:
        raise NotImplementedError("Pending Implementation")

    def clear_cache(self, refill: bool = True) -> PrairieLink:
        """
        Clears all cached values (except the prairieview version)

        :param refill: whether to refill the cache with newly retrieved values
        """
        for function in dir(self):
            try:
                self.__getattribute__(function).cache_clear()  # Clear cache using duck typing
            except (TypeError, AttributeError):
                pass  # I am not cached
            else:
                if refill:
                    self.__getattribute__(function)()  # Refill Cache

    def command(self, command: str) -> bool:
        """
        Send a script command to prairieview

        :param command: command as a string
        :return: indication of successful command
        """
        return self._prairie_link.SendScriptCommands(command)

    def connect(self, password: str = None) -> PrairieLink:
        """
        Makes initial connection to PrairieView at the beginning of the session. Function takes password as an argument
        to avoid unnecessarily storing passwords within an instance

        :param password: password for prairieview
        """
        # Via Michael Fox (Sr. Software Engineer; PrairieView): this password is necessary to stop other services from
        # accidentally interfacing with the software while port scanning. Each user on the PC receives a password
        # (not just admin) in the bottom left corner of 'Edit Scripts' within the Tools menu labeled
        # "Remote Access Password" By default, it says 'NONE' by in reality the default is "0.0.0.0".
        # This password is distinct from the configuration password within the 'configuration' application.

        if not password:  #
            password = getpass("Enter PrairieView Remote Access Password")
        status = self._prairie_link.Connect(f"{self._ip_address}", password)
        if not status:
            raise ConnectionError("Unable to Connect to PrairieView Instance")

    def disconnect(self) -> PrairieLink:
        """
        Disconnect from PrairieView

        """
        self._prairie_link.Disconnect()

    def get_motor_position(self, axis: int, device: int) -> float:
        """
        Returns the position within the specified axis on the specified device. Axis and device are both zero-indexed.

        :param axis: requested axis (x, y, z)
        :param device: requested device
        :return: motor position
        """
        return self._prairie_link.GetMotorPosition(MotorAxes(axis).name, device)

    def get_image(self, channel: int) -> array:
        """
        Returns a two-dimensional array of the current image for the specified channel. If each pixel is sampled
        multiple times, the results are *summed* not averaged.

        :param channel: channel to retrieve image from
        :return: image
        """
        return self._prairie_link.GetImage_2(channel, self._pixels_per_line(), self._lines_per_frame())

    def get_state(self, key: str, index: str = None, subindex: str = None) -> str:
        """
        Returns the value for the specified state / substate

        :param key: prairieview state
        :param index: index for state
        :param subindex: subindex for state
        :return: associated value
        """
        self._prairie_link.GetState(key, index, subindex)

    def set_timeout_and_retries(self, timeout: int = 2000, retries: int = 5) -> PrairieLink:
        """
        This method sets the timeout and number of times Prairie Link will attempt to resend commands to Prairie View
        without acknowledgment. If Prairie Link has not received an acknowledgment from Prairie View within the timeout
         (in milliseconds), it will re-send the most recent command until it receives acknowledgment or reaches the
          maximum number of retries. If no acknowledgment is received after the maximum number of retries,
          Prairie Link will give up and return control to the calling program. Setting the timeout to 0 disables the
           timeout, and Prairie Link will wait until it receives acknowledgment from Prairie View no matter how
           long it takes. For most uses it is not necessary to set these values, as the default values of 2000
           milliseconds and 5 retries are appropriate for all but very long commands (e.g. MarkPoints or
           SetCustomOutput scripts with many values).

        :param timeout: length of timeout in milliseconds
        :param retries: number of times to retry
        :return:
        """
        self._prairie_link.SetTimeoutAndRetries(timeout, retries)

    def unbuffered_raw_read(self) -> Any:
        """
        Reads a single raw data sample (unbuffered)

        :return: raw data
        """
        return self._prairie_link.ReadRawDataStream(self.samples_per_pixel)

    @cached(max_size=1)
    def _lines_per_frame(self) -> int:
        """
        Cached implementation for reading lines per frame, permitting more performant calls to read raw data stream

        :return: lines per frame
        """
        return self._prairie_link.LinesPerFrame()

    @cached(max_size=1)
    def _pixels_per_line(self) -> int:
        """
        Cached implementation for reading pixels per line, permitting more performant calls to read raw data stream

        :return: pixels per line
        """
        return self._prairie_link.PixelsPerLine()

    @cached(max_size=1)
    def _samples_per_pixel(self) -> int:
        """
        Cached implementation for reading samples per pixel, permitting more performant calls to read raw data stream

        :return: number of samples per pixel
        """
        return self._prairie_link.SamplesPerPixel()

    def __enter__(self) -> PrairieLink:
        """
        If using a context manager we will connect upon entry

        """
        if not self.connected:
            self.connect()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """
        PrairieView needs to perform some cleanup methods following disconnect or undesirable behavior could occur, so
        we will make sure to disconnect on exit from a context manager

        """
        self.disconnect()

    def __del__(self) -> None:
        """
        PrairieView needs to perform some cleanup methods following disconnect or undesirable behavior could occur

        """
        self.disconnect()


class MotorAxes(Enum):
    """
    Axis - Integer mapping for motor positions
    """
    X = 0
    Y = 1
    Z = 2
