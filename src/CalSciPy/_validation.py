from __future__ import annotations
import pathlib
from functools import wraps
import string
from typing import Callable, Tuple
from ._parsing import parameterize, amend_args
from ._style import TerminalStyle
from os import path
from os.path import exists


@parameterize
def validate_exists(function: Callable, pos: int = 0) -> Callable:
    """
    Decorator for validating existence of paths

    :param function: function to be decorated
    :type function: Callable
    :param pos: index of the argument to be validated
    :type pos: int
    """
    @wraps(function)
    def decorator(*args, **kwargs):
        string_input = str(args[pos])
        if not exists(string_input):
            raise FileNotFoundError(f"{TerminalStyle.GREEN}Invalid Path: {TerminalStyle.RESET}"
                             f"{TerminalStyle.YELLOW} Could not locate {TerminalStyle.RESET} "
                                    f"{TerminalStyle.BLUE}{string_input}{TerminalStyle.RESET}")
        # noinspection PyArgumentList
        return function(*args, **kwargs)
    return decorator


@parameterize
def validate_extension(function: Callable, required_extension: str, pos: int = 0) -> Callable:
    """
    Decorator for validating extension requirements

    :param function: function to be decorated
    :type function: Callable
    :param required_extension: required extension
    :type required_extension: str
    :param pos: index of the argument to be validated
    :type pos: int
    """
    @wraps(function)
    def decorator(*args, **kwargs):
        if not pathlib.Path(args[pos]).suffix:
            args = amend_args(args,  "".join([str(args[pos]), required_extension]), pos)
        if pathlib.Path(args[pos]).suffix != required_extension:
            raise ValueError(f"{TerminalStyle.GREEN}Input {pos}: {TerminalStyle.RESET}{TerminalStyle.YELLOW}"
                             f"filepath must contain the required extension {TerminalStyle.RESET}{TerminalStyle.BLUE}"
                             f"{required_extension}{TerminalStyle.RESET}")
        # noinspection PyArgumentList
        return function(*args, **kwargs)
    return decorator


@parameterize
def validate_filename(function: Callable, pos: int = 0) -> Callable:
    """
    Decorator for validating filenames

    :param function: function to be decorated
    :type function: Callable
    :param pos: index of the argument to be validated
    :type pos: int
    """
    @wraps(function)
    def decorator(*args, **kwargs):
        string_input = str(args[pos]).split("\\")[-1]
        if not set(string_input) <= set(string.ascii_letters + string.digits + "." + "_"):
            raise ValueError(f"{TerminalStyle.GREEN}Invalid Filename: {TerminalStyle.RESET}"
                             f"{TerminalStyle.YELLOW}Filenames are limited to standard letters and digits only."
                             f"{TerminalStyle.RESET}")
        # noinspection PyArgumentList
        return function(*args, **kwargs)
    return decorator


@parameterize
def validate_path(function: Callable, pos: int = 0) -> Callable:
    """
    Decorator for validating paths

    :param function: function to be decorated
    :type function: Callable
    :param pos: index of the argument to be validated
    :type pos: int
    """
    @wraps(function)
    def decorator(*args, **kwargs):
        string_input = str(args[pos])
        if [_char for _char in list(string_input) if _char is ":"].__len__() != 1:
            raise ValueError(f"{TerminalStyle.GREEN}Invalid Path: {TerminalStyle.RESET}"
                             f"{TerminalStyle.YELLOW}No root detected: "
                             f"{TerminalStyle.RESET}{TerminalStyle.GREEN}{string_input}{TerminalStyle.RESET}")
        if not set(string_input) <= set(string.ascii_letters + string.digits + "." + "\\" + ":" + "-" + "_"):
            raise ValueError(f"{TerminalStyle.GREEN}Invalid Path: {TerminalStyle.RESET}"
                             f"{TerminalStyle.YELLOW}Filenames are limited to standard letters, digits, backslash, "
                             f"colon, hyphen, and underscore only."
                             f"{TerminalStyle.RESET}")
        # noinspection PyArgumentList
        return function(*args, **kwargs)
    return decorator

