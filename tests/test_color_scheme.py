from __future__ import annotations

import pytest

# noinspection PyProtectedMember
from CalSciPy.color_scheme import _ColorScheme, _TerminalScheme, COLORS, TERM_SCHEME


def test_color_scheme():
    # check properties
    for attr in ["mapping", "num_colors"]:
        getattr(COLORS, attr)

    # first make sure that each color is unique and can be retrieved by str
    values = [COLORS(color_key) for color_key in dir(COLORS) if color_key.isupper()]
    unique_values = {value for value in values}
    assert(len(unique_values) == len(values))

    # next make sure that if we exceed the number of colors it repeats
    repeater_index = 0
    for i in range(COLORS.num_colors * 2):
        if i >= COLORS.num_colors:
            called_color = COLORS(i)
            assert(called_color == COLORS(repeater_index))
            repeater_index += 1
        else:
            called_color = COLORS(i)

    # finally check singleton status
    new_color_scheme = _ColorScheme()
    assert(new_color_scheme.__repr__() == COLORS.__repr__())


def test_terminal_scheme():

    # check properties
    for attr in ["type", "emphasis", "header"]:
        getattr(TERM_SCHEME, attr)

    # check all styles unique
    keys = [key for key in dir(TERM_SCHEME) if key.isupper()]
    assert (len(keys) == len({getattr(TERM_SCHEME, key) for key in keys}))

    # check wrapping messages actually resets
    new_msg = TERM_SCHEME("42!", "type")
    msg_parts = new_msg.split("!")
    assert(msg_parts[-1] == "\x1b[0m")

    # check msg still delivered if failed style request
    new_msg = TERM_SCHEME("42!", "Adams")
    assert("42!" in new_msg)

    # finally check singleton status
    new_terminal_scheme = _TerminalScheme()
    assert(new_terminal_scheme.__repr__() == TERM_SCHEME.__repr__())
