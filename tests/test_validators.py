import pytest

# noinspection PyProtectedMember
from CalSciPy._validators import collector, validate_extension, validate_matrix, validate_tensor


"""
Tests of validators


"""


def test_collector():

    # used kwargs
    args = None,
    kwargs = {"dummy": "dummy"}
    collected, target, use_args = collector(0, "dummy", *args, **kwargs)
    assert collected
    assert not use_args
    assert (target == "dummy")

    # used args
    args = ("dummy", "variable")
    kwargs = {}
    collected, target, use_args = collector(0, "dummy", *args, **kwargs)
    assert collected
    assert use_args
    assert (target == "dummy")

    # failure
    args = None,
    kwargs = {}
    collected, target, use_args = collector(0, "dummy", *args, **kwargs)
    assert not collected
    assert not use_args
    assert not target


def test_validate_extension():
    # generate_decorated function
    @validate_extension(required_extension=".marvin", pos=0, key="a")
    def valid_handle(a, b):
        return 0

    # test valid
    valid_handle("C:\\the_paranoid_android.marvin", None)
    # test invalid
    with pytest.raises(ValueError):
        valid_handle("C:\\the_paranoid_android.arthur", None)


def test_validate_matrix(sample_matrix, sample_tensor):
    # generate_decorated function
    @validate_matrix(pos=0, key="a")
    def valid_handle(a, b):
        return 0

    # test valid
    valid_handle(sample_matrix, sample_tensor)
    # test invalid
    with pytest.raises(AssertionError):
        valid_handle(sample_tensor, sample_matrix)


def test_validate_tensor(sample_matrix, sample_tensor):
    # generate_decorated function
    @validate_tensor(pos=1, key="b")
    def valid_handle(a, b):
        return 0

    # test valid
    valid_handle(sample_matrix, sample_tensor)
    # test invalid
    with pytest.raises(AssertionError):
        valid_handle(sample_tensor, sample_matrix)
