from __future__ import annotations

import pathlib
from functools import wraps
import string
from typing import Callable, Tuple


def parameterize(decorator: Callable) -> Callable:
    """
    Function for decorating decorators with parameters

    Based on -> https://stackoverflow.com/questions/46734219/flask-error-with-two-parameterized-functions

    :param decorator: a decorator
    :type decorator: Callable
    """

    def outer(*args, **kwargs):

        def inner(func):
            # noinspection PyArgumentList
            return decorator(func, *args, **kwargs)

        return inner

    return outer


@parameterize
def validate_filename(function: Callable, pos: int = 0) -> Callable:
    """
    Decorator for validating filenames

    :param function: function to be decorated
    :type function: Callable
    :param pos: index of the filename to be validated
    :type pos: int
    """
    @wraps(function)
    def decorator(*args, **kwargs):
        string_input = str(args[pos])
        if not set(string_input) <= set(string.ascii_letters + string.digits):
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
    :param pos: index of the path to be validated
    :type pos: int
    """
    @wraps(function)
    def decorator(*args, **kwargs):
        string_input = str(args[pos])
        if [_char for _char in list(string_input) if _char is ":"].__len__() != 1:
            raise ValueError(f"{terminal_style.GREEN}Invalid Path: {terminal_style.RESET}"
                             f"{terminal_style.YELLOW}No root detected."
                             f"{terminal_style.RESET}")
        if not set(string_input) <= set(string.ascii_letters + string.digits + "." + "\\" + ":" + "-" + "_"):
            raise ValueError(f"{terminal_style.GREEN}Invalid Path: {terminal_style.RESET}"
                             f"{terminal_style.YELLOW}Filenames are limited to standard letters, digits, backslash, "
                             f"colon, hyphen, and underscore only."
                             f"{terminal_style.RESET}")
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
    :param pos: index of the path to be validated
    :type pos: int
    """
    @wraps(function)
    def decorator(*args, **kwargs):
        if not pathlib.Path(args[pos]).suffix:
            args = list(args)
            args[pos] = "".join([str(args[pos]), required_extension])
            args = tuple(args)
        if pathlib.Path(args[pos]).suffix != required_extension:
            raise ValueError(f"{terminal_style.GREEN}Input {pos}: {terminal_style.RESET}{terminal_style.YELLOW}"
                            f"filepath must contain the required extension {terminal_style.RESET}{terminal_style.BLUE}"
                             f"{required_extension}{terminal_style.RESET}")
        # noinspection PyArgumentList
        return function(*args, **kwargs)
    return decorator


@parameterize
def convert_optionals(function: Callable, allowed, required, pos: int = 0) -> Callable:
    @wraps(function)
    def decorator(*args, **kwargs):
        allowed_input = args[pos]
        if isinstance(allowed_input, allowed):
            allowed_input = required(allowed_input)
        if not isinstance(allowed_input, required):
            raise TypeError(f"{terminal_style.GREEN}Input {pos}: {terminal_style.RESET}{terminal_style.YELLOW}"
                            f"inputs are permitted to be of the following types {terminal_style.RESET}"
                            f"{terminal_style.BLUE}{allowed}{terminal_style.RESET}")
        args = list(args)
        args[pos] = allowed_input
        args = tuple(args)
        # noinspection PyArgumentList
        return function(*args, **kwargs)
    return decorator


class terminal_style:
    RED = "\u001b[31m"
    GREEN = "\u001b[32m"
    BLUE = "\u001b[34m"
    YELLOW = "\u001b[38;5;11m"
    BOLD = "\u001b[1m"
    UNDERLINE = "\u001b[7m"
    RESET = "\033[0m"
