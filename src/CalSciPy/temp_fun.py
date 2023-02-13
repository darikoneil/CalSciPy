from PPVD.parsing import parameterize
from PPVD.style import TerminalStyle
from typing import Callable, Tuple


@parameterize
def validate_longest_numpy_dimension(function: Callable, axis: int = 0, pos: int = 0) -> Callable:
    """
    Decorator for validating longest dimension of a numpy array
    (e.g., to ensure a video is organized in frame x height x width format)

    :param function: function to be decorated
    :type function: Callable
    :param axis: axis to be assert is the longest
    :type axis: int
    :param pos: index of the argument to be validated
    :type pos: int
    """

    @wraps(function)
    def decorator(*args, **kwargs):
        arg_shape = args[pos]
        long_axis = arg_shape[axis]
        axes_list = list(arg_shape)
        axes_list.pop(axis)

        for _axis in axes_list:
            if _axis > long_axis:
                raise AssertionError(f"{TerminaLStyle.GREEN}Input {pos} Improper Format: "
                                     f"{TerminalStyle.YELLOW} axis {TerminalStyle.BLUE}{axis}"
                                     f"{TerminalStyle.YELLOW} ought to be larger than axis "
                                     f"{TerminalStyle.BLUE}{_axis}{TerminalStyle.RESET}")
        # noinspection PyArgumentList
        return function(*args, **kwargs)

    return decorator


@parameterize
def validate_numpy_dimension_odd(function: Callable, odd_dimensions: Tuple[int] = tuple(0), pos: int = 0) -> Callable:
    """
    Decorator for validating numpy dimension is odd

    :param function: function to be decorated
    :type function: Callable
    :param odd_dimensions: which dimensions odd_dimensions
    :type odd_dimensions: tuple[int]
    :param pos: index of the argument to be validated
    :type pos: int
    """
    @wraps(function)
    def decorator(*args, **kwargs):
        for _dim in odd_dimensions:
            if args[pos].shape[_dim] % 2 != 0:
                raise TypeError(f"{TerminalStyle.GREEN}Input {pos} Improper Format"
                            f"{TerminalStyle.YELLOW}the dimension "
                            f"{TerminalStyle.BLUE}{_dim}{TerminalStyle.YELLOW}"
                            f" must be odd{TerminalStyle.RESET}")
        # noinspection PyArgumentList
        return function(*args, **kwargs)
    return decorator
