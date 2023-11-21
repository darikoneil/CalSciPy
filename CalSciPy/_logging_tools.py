from __future__ import annotations
from typing import Any, Iterable
from datetime import datetime
from IPython import get_ipython
from collections import deque
from pathlib import Path
from PPVD.parsing import convert_permitted_types_to_required


DEFAULT_PATH = Path.cwd()


class IPythonLogger:
    def __init__(self, directory: Path = DEFAULT_PATH):
        """
        Logging class

        :param path: path to save file
        :type path: str
        """

        #: object: IPython magic
        self._IP = None
        #: pathlib.Path: path to log file
        self._log_file = directory.joinpath("log_file.log")

        if directory.exists() and not self._log_file.exists():
            self.create_log()

        self.start_log()

    def check_log_status(self) -> IPythonLogger:
        """
        Checks log status

        :rtype: IPythonLogger
        """

        self._IP.run_line_magic('logstate', '')

    def create_log(self) -> IPythonLogger:
        """
        Creates a log file for a new instance

        :rtype: IPythonLogger
        """
        # do this to physically reserve the file
        with open(self._log_file, "w") as log:
            log.write("")

    def pause_log(self) -> IPythonLogger:
        """
        pause Logging

        :rtype: IPythonLogger
        """
        self._IP.run_line_magic('logstop', '')

    def end_log(self) -> IPythonLogger:
        """
        end log

        :rtype: IPythonLogger
        """
        self.pause_log()
        self._IP = None

    def start_log(self) -> IPythonLogger:
        """
        Starts Log

        :rtype: IPythonLogger
        """
        self._IP = get_ipython()
        _magic_arguments = '-o -r -t ' + str(self._log_file) + ' append'
        self._IP.run_line_magic('logstart', _magic_arguments)
        print("Logging Initiated")

    def __json_decode__(self, **attrs):
        directory = attrs.get("_log_file").parent
        self.__init__(directory)


class ModificationLogger(deque):

    def append(self, __x: Any) -> ModificationLogger:
        __x = (__x, get_timestamp())
        super().append(__x)

    def appendleft(self, __x: Any) -> ModificationLogger:
        __x = (__x, get_timestamp())
        super().appendleft(__x)

    def extend(self, __iterable: Iterable[Any]) -> ModificationLogger:
        for iter_ in __iterable:
            iter_ = (iter_, get_timestamp())
            self.append(iter_)

    def extendleft(self, __iterable: Iterable[Any]) -> ModificationLogger:
        for iter_ in __iterable:
            iter_ = (iter_, get_timestamp())
            self.appendleft(iter_)

    def load(self, value: Any) -> ModificationLogger:
        super().appendleft(value)

    def __json_encode__(self):
        return {"values": list(self)}

    def __json_decode__(self, **attrs):
        for value in attrs.get("values"):
            self.load(tuple(value))


def get_timestamp() -> str:
    """
    Uses datetime to return date/time str. Simply a function because I'll remember this and not that

    :return: current date
    :rtype: str
    """
    return datetime.now().strftime("%m-%d-%Y %H:%M:%S")


@convert_permitted_types_to_required(permitted=(str, Path), required=str, pos=0)
def generate_read_me(path: str, text: str) -> None:
    """
    Generate a read me file

    :param path: path to save readme
    :type path: str
    :param text: text to write in readme file
    :type text: str
    :rtype: None
    """
    with open(path, 'w') as file:
        file.write(text)
