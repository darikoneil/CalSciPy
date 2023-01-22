from __future__ import annotations
import pathlib
from functools import wraps
import string
from typing import Callable, Tuple
from ._parsing import parameterize, amend_args
from ._style import terminal_style
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
            raise FileNotFoundError(f"{terminal_style.GREEN}Invalid Path: {terminal_style.RESET}"
                             f"{terminal_style.YELLOW} Could not locate {terminal_style.RESET} "
                                    f"{terminal_style.BLUE}{string_input}{terminal_style.RESET}")
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
            raise ValueError(f"{terminal_style.GREEN}Input {pos}: {terminal_style.RESET}{terminal_style.YELLOW}"
                             f"filepath must contain the required extension {terminal_style.RESET}{terminal_style.BLUE}"
                             f"{required_extension}{terminal_style.RESET}")
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
            raise ValueError(f"{terminal_style.GREEN}Invalid Filename: {terminal_style.RESET}"
                             f"{terminal_style.YELLOW}Filenames are limited to standard letters and digits only."
                             f"{terminal_style.RESET}")
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
            raise ValueError(f"{terminal_style.GREEN}Invalid Path: {terminal_style.RESET}"
                             f"{terminal_style.YELLOW}No root detected: "
                             f"{terminal_style.RESET}{terminal_style.GREEN}{string_input}{terminal_style.RESET}")
        if not set(string_input) <= set(string.ascii_letters + string.digits + "." + "\\" + ":" + "-" + "_"):
            raise ValueError(f"{terminal_style.GREEN}Invalid Path: {terminal_style.RESET}"
                             f"{terminal_style.YELLOW}Filenames are limited to standard letters, digits, backslash, "
                             f"colon, hyphen, and underscore only."
                             f"{terminal_style.RESET}")
        # noinspection PyArgumentList
        return function(*args, **kwargs)
    return decorator

