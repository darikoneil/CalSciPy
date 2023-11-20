from __future__ import annotations
from typing import List, Tuple, Any
from PPVD.style import TerminalStyle
from dataclasses import Field
from types import MappingProxyType
import re


def _validate_range(var: Any, val_range: Tuple[Any, Any]) -> List[Exception]:
    dingus = DingusLogger()
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
                dingus += e
        return dingus.exceptions


def _validate_value_length(var: Any, val_length: int) -> Exception:
    dingus = DingusLogger()
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
                dingus += e
        return dingus.exceptions


def _validate_length(var: Any, val_length: int) -> Exception:
    dingus = DingusLogger()
    try:
        _ = iter(var)
    except TypeError:
        if len(var) != val_length:
            return [AssertionError("Value Length")]
    else:
        return dingus.exceptions


class DingusLogger:
    def __init__(self, exceptions: List[Exception] = None):
        self.exceptions = []
        if exceptions:
            self.add_exception(exceptions)

    def __str__(self):
        message = f"{TerminalStyle.YELLOW}"
        for exception in self.exceptions:
            message += f"\n{exception}"
        message += f"{TerminalStyle.RESET}"
        return message

    def add_exception(self, other: Exception or List[Exception]) -> DingusLogger:
        try:
            _ = iter(other)
        except TypeError:
            self.exceptions.append(other)
        else:
            for exception in other:
                self.add_exception(exception)
        self.exceptions = [exception for exception in self.exceptions if exception is not None]

    def raise_exceptions(self) -> DingusLogger:
        raise DingusException(self.__str__())

    def __add__(self, other: Exception or List[Exception]):
        try:
            _ = iter(other)
        except TypeError:
            self.add_exception(other)
        else:
            for exception in other:
                self.add_exception(exception)
        return DingusLogger(self.exceptions)

    def __call__(self, *args, **kwargs):
        self.raise_exceptions()


class DingusException(Exception):
    def __init__(self, message: str):
        super().__init__(message)
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


FIELD_VALIDATORS = MappingProxyType(dict({
    "range": _validate_range,
    "length": _validate_length,
    "value_length": _validate_value_length,
}))


def field_validator(key: str, value: Any, var: Field) -> List[Exception]:
    if value is None:
        return
    dingus = DingusLogger()
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
        dingus += e
        return dingus.exceptions  # short-circuit to prevent shenanigans on the field validators which are type specific

    # now use validators on metadata
    meta = var.metadata

    for key in meta.keys():
        if key in FIELD_VALIDATORS.keys():
            e = FIELD_VALIDATORS.get(key)(value, meta.get(key))
            if e:
                dingus += e
    return dingus.exceptions


def format_fields(var: object) -> tuple:
    return tuple([(key, var.__dict__.get(key), var.__dataclass_fields__.get(key))
                  for key in sorted(var.__dataclass_fields__)])


def validate_fields(data_class: object) -> bool:
    dingus = DingusLogger()
    fields_tuple = format_fields(data_class)
    for key_, value_, field_ in fields_tuple:
        dingus.add_exception(field_validator(key=key_, value=value_, var=field_))

    if dingus.exceptions:
        dingus.raise_exceptions()
