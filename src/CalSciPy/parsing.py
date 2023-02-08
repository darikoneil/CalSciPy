from __future__ import annotations
import os.path
from typing import Callable, Tuple, Any, List
from functools import wraps
from .style import TerminalStyle
from os import getcwd, path
from os.path import isdir, isabs


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
def convert_permitted_types_to_required(function: Callable, permitted: Tuple, required: Any, pos: int = 0) -> Callable:
    """
    Decorator that converts a tuple of permitted types to type supported by the decorated method

    :param function: function to be decorated
    :type function: Callable
    :param permitted: permitted types
    :type permitted: Tuple
    :param required: type required by code
    :type required: Any
    :param pos: index of argument to be converted
    :type pos: int
    """
    @wraps(function)
    def decorator(*args, **kwargs):
        allowed_input = args[pos]
        if isinstance(allowed_input, permitted):
            allowed_input = required(allowed_input)
        if not isinstance(allowed_input, required):
            raise TypeError(f"{TerminalStyle.GREEN}Input {pos}: {TerminalStyle.YELLOW}"
                            f"inputs are permitted to be of the following types "
                            f"{TerminalStyle.BLUE}{permitted} {TerminalStyle.RESET}")
        args = amend_args(args, allowed_input, pos)
        # noinspection PyArgumentList
        return function(*args, **kwargs)
    return decorator


@parameterize
def if_dir_join_filename(function: Callable, default_name: str, flag_pos: int = 0) -> Callable:
    """
    Decorator that joins a default filename if a directory is passed

    :param function: function to be decorated
    :type function: Callable
    :param default_name: default filename to be joined to flag_pos
    :type default_name: str
    :param flag_pos: index of argument serving as flag
    :type flag_pos: int
    """
    @wraps(function)
    def decorator(*args, **kwargs):
        if isdir(args[flag_pos]):
            args = amend_args(args, "".join([args[flag_pos], "\\", default_name]), flag_pos)
        # noinspection PyArgumentList
        return function(*args, **kwargs)
    return decorator


@parameterize
def if_dir_append_filename(function: Callable, default_name: str, flag_pos: int = 0) -> Callable:
    """
     Decorator that generates a default filepath within a passed directory and appends to arguments

    :param function: function to be decorated
    :type function: Callable
    :param default_name: default filename to be appended
    :type default_name: str
    :param flag_pos: index of argument serving as flag
    :type flag_pos: int
    """
    @wraps(function)
    def decorator(*args, **kwargs):
        if isdir(args[flag_pos]):
            args = append_args(args, "".join([args[flag_pos], "\\", default_name]))
        # noinspection PyArgumentList
        return function(*args, **kwargs)
    return decorator


@parameterize
def require_full_path(function: Callable, pos: int = 0) -> Callable:
    """
    Decorator to determine if an argument is a filename, and if so generate full path

    :param function: function to be decorated
    :type function: Callable
    :param pos: index of argument to be converted
    :type pos: int
    """
    @wraps(function)
    def decorator(*args, **kwargs):
        _input = str(args[pos])
        if isdir(_input):
            args = amend_args(args, "".join([getcwd(), "\\", _input]), pos)
        # noinspection PyArgumentList
        return function(*args, **kwargs)
    return decorator


def amend_args(arguments: Tuple, amendment: Any, pos: int = 0) -> Tuple:
    """
    Function amends arguments tuple (~scary tuple mutation~)

    :param arguments: arguments to be amended
    :type arguments: Tuple
    :param amendment: new value of argument
    :type amendment: Any
    :param pos: index of argument to be converted
    :type pos: int
    :return: amended arguments
    :rtype: Tuple
    """

    arguments = list(arguments)
    arguments[pos] = amendment
    return tuple(arguments)


def append_args(arguments: Tuple, value: Any) -> Tuple:
    """
    Function appends arguments tuple (~scary tuple mutation~)

    :param arguments: arguments to be appended
    :type arguments: Tuple
    :param value: new value of argument
    :type value: Any
    :return: appended arguments
    :rtype: Tuple
    """
    arguments = list(arguments)
    arguments.append(value)
    return tuple(arguments)


def check_split_strings(tag_: str, str_to_split: str) -> str:
    return [_tag for _tag in str_to_split.split("_") if tag_ in _tag]
# REFACTOR


def find_num_unique_files_given_static_substring(tag: str, files: List[pathlib.Path]) -> int:
    _hits = [check_split_strings(tag, str(_file)) for _file in files]
    _hits = [_hit for _nested_hit in _hits for _hit in _nested_hit]
    return list(_hits).__len__()
# REFACTOR


def find_num_unique_files_containing_tag(tag: str, files: List[pathlib.Path]) -> int:
    _hits = [check_split_strings(tag, str(_file)) for _file in files]
    _hits = [_hit for _nested_hit in _hits for _hit in _nested_hit]
    return list(dict.fromkeys(_hits)).__len__()
# REFACTOR
