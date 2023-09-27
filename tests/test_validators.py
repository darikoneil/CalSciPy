import pytest


from CalSciPy._validators import validate_extension


"""
Tests of validators


"""


def dummy_function(arg_zero=0, arg_one=1, arg_two=2):
    return 0


def test_validate_extension():
    # generate_decorated function
    valid_handle = validate_extension(dummy_function, ".marvin", 0, "arg_zero")
    # test valid
    # valid_handle("C:\\the_paranoid_android.marvin")
    # test invalid
    # with pytest.raises(ValueError):
    #    valid_handle("C:\\the_paranoid_android.arthur")
