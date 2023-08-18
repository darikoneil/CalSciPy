from __future__ import annotations

import pytest

# noinspection PyProtectedMember
from CalSciPy.color_scheme import _ColorScheme, _TerminalScheme, COLORS, TERM_SCHEME


def test_color_scheme():

    # first make sure that each color is unique and can be retrieved by str
    values = [COLORS(color_key) for color_key in COLORS.colors]
    unique_values = {value for value in values}
    assert(len(unique_values) == len(values))

    # next make sure that if we exceed the number of colors it repeats
    repeater_index = 0
    for i in range(COLORS.num_colors + 2):
        if i > COLORS.num_colors:
            called_color = COLORS(i)
            assert(called_color == COLORS(repeater_index))
            repeater_index += 1

    # finally check singleton status
    new_color_scheme = _ColorScheme()
    assert(new_color_scheme.__repr__() == COLORS.__repr__())
