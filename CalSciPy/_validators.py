from __future__ import annotations
from typing import Callable, Tuple, Any, Union, List
from types import MappingProxyType
from functools import wraps
from pathlib import Path
import string
import re

from .color_scheme import TERM_SCHEME

# backport if necessary
from sys import version_info
if version_info.minor < 10:
    from _backports import dataclass, Field
else:
    from dataclasses import dataclass, Field


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


def validate_field_length(var: Any, val_length: int) -> Exception:
    logger = MultiExceptionLogger()
    try:
        _ = iter(var)
    except TypeError:
        if len(var) != val_length:
            return [AssertionError("Value Length")]
    else:
        return logger.exceptions


def validate_field_range(var: Any, val_range: Tuple[Any, Any]) -> List[Exception]:
    logger = MultiExceptionLogger()
    val_min, val_max = val_range
    try:
        _ = iter(var)
    except TypeError:
        if not val_min <= var <= val_max:
            return [AssertionError("Range")]
    else:
        for val in var:
            try:
                assert (val_min <= val <= val_max), "Range"
            except AssertionError as e:
                logger += e
        return logger.exceptions


def validate_field_value_length(var: Any, val_length: int) -> Exception:
    logger = MultiExceptionLogger()
    for val in var.values():
        try:
            _ = iter(val)
        except TypeError:
            if val_length != 1:
                return [AssertionError("Value Length")]
        else:
            try:
                assert (len(val) == val_length)
            except AssertionError:
                e = AssertionError("Value Length")
                logger += e
        return logger.exceptions


def validate_keys(element: Any, parameters: dict) -> dict:
    if isinstance(parameters, dict):
        expected_keys = vars(element).keys()
        return {key: value for key, value in parameters.items() if key in expected_keys}
    else:
        return {}


FIELD_VALIDATORS = MappingProxyType(dict({
    "length": validate_field_length,
    "range": validate_field_range,
    "value_length": validate_field_value_length

}))


class MultiExceptionLogger:
    def __init__(self, exception_class: Exception = Exception, exceptions: List[Exception] = None):
        self.exceptions = []
        self.exception_class = exception_class
        if exceptions:
            self.add_exception(exceptions)

    def __str__(self):
        exceptions = "".join([f"\n{message}" for message in self.exceptions])
        return TERM_SCHEME(*exceptions, "emphasis")

    def add_exception(self, other: Union[Exception, List[Exception]]) -> MultiExceptionLogger:
        try:
            _ = iter(other)
        except TypeError:
            self.exceptions.append(other)
        else:
            for exception in other:
                self.add_exception(exception)
        self.exceptions = [exception for exception in self.exceptions if exception is not None]

    def raise_exceptions(self) -> MultiExceptionLogger:
        # noinspection PyCallingNonCallable
        raise self.exception_class(self.__str__())

    def __add__(self, other: Union[Exception, List[Exception]]):
        try:
            _ = iter(other)
        except TypeError:
            self.add_exception(other)
        else:
            for exception in other:
                self.add_exception(exception)
        return MultiExceptionLogger(self.exceptions)

    def __call__(self, *args, **kwargs):
        self.raise_exceptions()


class MultiException(Exception):
    def __init__(self, errors: List[Exception]):
        self.errors = errors
        super().__init__(self.errors)
        return


def type_check_nested_types(var: Any, expected: str) -> bool:
    """
    Checks type of nested types. WORKS FOR ONLY ONE NEST.

    :param var: variable to check
    :type var: Any
    :param expected: expected type
    :type expected: str
    :return: boolean type comparison
    :rtype: bool
    """
    # noinspection DuplicatedCode
    try:
        if isinstance(var, (MappingProxyType, dict)):
            pass
        _ = iter(var)
    except TypeError:
        return isinstance(var, eval(expected))
    else:
        # separate out the nested types
        expected = re.split(r"[\[\],]", expected)[:-1]
        outer_type = isinstance(var, eval(expected[0]))
        expected.pop(0)

        var_list = list(var)

        # if the provided types aren't equal to number of items in var, then fill out with last type
        if len(var_list) != len(expected):
            while len(var_list) != len(expected):
                expected.append(expected[-1])
        try:
            assert (len(var.keys()) >= 1)
            checks = [isinstance(nested_var, eval(nested_type))
                      for nested_var, nested_type in zip(var.values(), expected)]
            checks.append(outer_type)
        except AttributeError:
            checks = [isinstance(nested_var, eval(nested_type))
                      for nested_var, nested_type in zip(var, expected)]
            checks.append(outer_type)
        return all(checks)


def format_fields(data_class: dataclass()) -> Tuple[Tuple[str, Any, Field], ...]:
    return tuple([(key, data_class.__dict__.get(key), data_class.__dataclass_fields__.get(key))
                  for key in sorted(data_class.__dataclass_fields__)])


def field_validator(key: str, value: Any, var: Field) -> List[Exception]:
    if value is None:
        return

    logger = MultiExceptionLogger()
    # noinspection DuplicatedCode
    type_ = var.type

    # first check type always
    try:
        # Type Check
        if not isinstance(value, eval(type_)):
            raise AttributeError(f"Field {key} must be type {type_} not {type(value).__name__}")
    except TypeError:
        # Type Check
        if not type_check_nested_types(value, str(type_)):
            raise AttributeError(f"Field {key} must be type {type_} not {type(value).__name__}")
    except AttributeError as e:
        logger += e
        return logger.exceptions  # short-circuit to prevent shenanigans on the field validators which are type specific

    # now use validators on metadata
    meta = var.metadata

    for key in meta.keys():
        if key in FIELD_VALIDATORS.keys():
            e = FIELD_VALIDATORS.get(key)(value, meta.get(key))
            if e:
                logger += e
    return logger.exceptions


def validate_fields(data_class: dataclass) -> bool:
    logger = MultiExceptionLogger()
    fields = format_fields(data_class)

    for key, val, field_ in fields:
        logger.add_exception(field_validator(key=key, value=val, var=field_))

    if logger.exceptions:
        logger.raise_exceptions()
