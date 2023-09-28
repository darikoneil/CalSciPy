from __future__ import annotations
from typing import Callable, Tuple, Any
from functools import wraps
from pathlib import Path
import string


def collector(pos: int, key: str, *args, **kwargs) -> Tuple[bool, Any, bool]:
    # noinspection PyBroadException
    try:
        if key in kwargs:
            collected = True
            use_args = False
            target = kwargs.get(key)
        elif pos is not None and args[pos] is not None:
            collected = True
            use_args = True
            target = args[pos]
        else:
            raise Exception
    except Exception:  # if any exception, just report a failure to collect
        collected = False
        use_args = None
        target = None
    # noinspection PyUnboundLocalVariable
    return collected, target, use_args


def parameterize(decorator: Callable) -> Callable:
    """
    Function for decorating decorators with parameters

    :param decorator: a decorator

    :type decorator: Callable
    """

    def outer(*args, **kwargs) -> Callable:

        def inner(func: Callable) -> Callable:
            # noinspection PyArgumentList
            return decorator(func, *args, **kwargs)

        return inner

    return outer


@parameterize
def convert_permitted_types_to_required(function: Callable,
                                        permitted: Tuple,
                                        required: Any,
                                        pos: int = 0,
                                        key: str = None) -> Callable:
    """
    Decorator that converts a tuple of permitted types to type supported by the decorated method

    :param function: function to be decorated
    :type function: Callable
    :param permitted: permitted types
    :type permitted: tuple
    :param required: type required by code
    :type required: Any
    :param pos: index of argument to be converted
    :type pos: int
    :param key: keyword of argument to be converted
    :type key: str
    """
    @wraps(function)
    def decorator(*args, **kwargs) -> Callable:

        collected, allowed_input, use_args = collector(pos, key, *args, **kwargs)

        if collected:
            if isinstance(allowed_input, permitted):
                allowed_input = required(allowed_input)

            if not isinstance(allowed_input, required):
                raise TypeError(f"{pos}, {key}")

            if use_args:
                args = amend_args(args, allowed_input, pos)
            else:
                kwargs[key] = allowed_input

        return function(*args, **kwargs)

    return decorator


@parameterize
def validate_extension(function: Callable, required_extension: str, pos: int = 0, key: str = None) \
        -> Callable:  # noqa: U100
    """
    Decorator for validating extension requirements

    :param function: function to be decorated
    :type function: Callable
    :param required_extension: required extension
    :type required_extension: str
    :param pos: index of the argument to be validated
    :type pos: int
    :param key: keyword of the argument to be validated
    """
    @wraps(function)
    def decorator(*args, **kwargs) -> Callable:
        if not Path(args[pos]).suffix:
            args = amend_args(args, "".join([str(args[pos]), required_extension]), pos)
        if Path(args[pos]).suffix != required_extension:
            raise ValueError(f"Input {pos}: filepath must contain the required extension {required_extension}")
        # noinspection PyArgumentList
        return function(*args, **kwargs)
    return decorator


@parameterize
def validate_matrix(function: Callable, pos: int = 0, key: str = None) -> Callable:
    """
    Decorator for validating matrices

    :param function: function to be decorated
    :type function: Callable
    :param pos: index of the argument to be validated
    :type pos: int
    :param key: keyword of the argument to be validated
    """
    @wraps(function)
    def decorator(*args, **kwargs) -> Callable:
        var_input = args[pos]
        if len(var_input.shape) != 2:
            raise AssertionError(f"Input {pos}, {key}: required to be in matrix format")
        # noinspection PyArgumentList
        return function(*args, **kwargs)
    return decorator


@parameterize
def validate_filename(function: Callable, pos: int = 0, key: str = None) -> Callable:  # noqa: U100
    """
    Decorator for validating filenames

    :param function: function to be decorated
    :type function: Callable
    :param pos: index of the argument to be validated
    :type pos: int
    :param key: keyword of the argument to be validated
    """
    @wraps(function)
    def decorator(*args, **kwargs) -> Callable:
        string_input = str(args[pos]).split("\\")[-1]
        if not set(string_input) <= set(string.ascii_letters + string.digits + "." + "_"):
            raise ValueError("Invalid Filename: filenames are limited to standard letters and digits only.")
        # noinspection PyArgumentList
        return function(*args, **kwargs)
    return decorator


def amend_args(arguments: Tuple, amendment: Any, pos: int = 0) -> Tuple:
    """
    Function amends arguments tuple (~scary tuple mutation~)

    :param arguments: arguments to be amended
    :type arguments: tuple
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


@parameterize
def validate_evenly_divisible(function: Callable, numerator: int = 0, denominator: int = 1, axis: int = 1) -> Callable:
    """
    Decorator for validating existence of paths

    :param function: function to be decorated
    :type function: Callable
    :param numerator: position of numerator for division
    :type numerator: int
    :param denominator: position of denominator for division
    :type denominator: int
    :param axis: axis of numerator to divide
    :type axis: int
    """
    @wraps(function)
    def decorator(*args, **kwargs) -> Callable:
        numerator_val = args[numerator]
        denominator_val = args[denominator]

        if numerator_val.shape[axis] // denominator_val != numerator_val.shape[axis] / denominator_val:
            raise AssertionError("error")
        # noinspection PyArgumentList
        return function(*args, **kwargs)
    return decorator


@parameterize
def validate_tensor(function: Callable, pos: int = 0, key: str = None) -> Callable:
    """
    Decorator to assert argument is a tensor

    :param function: function to be decorated
    :type function: Callable
    :param pos: index of the argument to be validated
    :type pos: int
    :param key: keyword of the argument to be validated
    """
    @wraps(function)
    def decorator(*args, **kwargs) -> Callable:
        var_input = args[pos]
        if len(var_input.shape) != 3:
            raise AssertionError(f"{pos}, {key}")
        # noinspection PyArgumentList
        return function(*args, **kwargs)
    return decorator
