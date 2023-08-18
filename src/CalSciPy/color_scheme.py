from __future__ import annotations
from typing import Tuple, Union
from operator import eq

from ._backports import PatternMatching


class _ColorScheme:
    """
    A container class for CalSciPy's color scheme

    """
    blue: Tuple[float, float, float] = (15 / 255, 159 / 255, 255 / 255)
    orange: Tuple[float, float, float] = (255 / 255, 159 / 255, 15 / 255)
    green: Tuple[float, float, float] = (64 / 255, 204 / 255, 139 / 255)
    red: Tuple[float, float, float] = (255 / 255, 78 / 255, 75 / 255)
    purple: Tuple[float, float, float] = (120 / 255, 64 / 255, 204 / 255)
    yellow: Tuple[float, float, float] = (255 / 255, 240 / 255, 15 / 255)
    shadow: Tuple[float, float, float] = (224 / 255, 224 / 255, 224 / 255)
    light: Tuple[float, float, float] = (192 / 255, 192 / 255, 192 / 255)
    medium: Tuple[float, float, float] = (128 / 255, 128 / 255, 128 / 255)
    dark: Tuple[float, float, float] = (65 / 255, 65 / 255, 65 / 255)
    black: Tuple[float, float, float] = (0 / 255, 0 / 255, 0 / 255)
    white: Tuple[float, float, float] = (255 / 255, 255 / 255, 255 / 255)
    background: Tuple[float, float, float] = (245 / 255, 245 / 255, 245 / 255)
    mapping = list(enumerate([red, green, blue, orange, purple, yellow, black, medium, dark, light]))

    def __new__(cls: _ColorScheme) -> _ColorScheme:
        """
        Force color scheme as singleton

        """
        if not hasattr(cls, "instance"):
            cls.instance = super(_ColorScheme, cls).__new__(cls)
        return cls.instance

    @property
    def colors(self) -> set:
        return {key for key in dir(COLORS) if "__" not in key and "instance" not in key and "mapping" not in key}

    @property
    def num_colors(self) -> int:
        return len(self.colors)

    def __call__(self, value: Union[int, str], *args, **kwargs) -> Tuple[float, float, float]:
        """
        Call for next color in scheme

        :param value: requested color directly or by index
        :returns: tuple of RGB values
        """
        if isinstance(value, int):
            if value >= self.num_colors:
                value -= self.num_colors * (value // self.num_colors)
            return self.mapping[value][1]
        elif isinstance(value, str):
            return getattr(self, value)


class _TerminalScheme:
    """
    A container class for CalSciPy's terminal printing color/font scheme

    """
    BLUE = "\u001b[38;5;39m"  # Types
    YELLOW = "\u001b[38;5;11m"  # Emphasis
    BOLD = "\u001b[1m"  # Titles, Headers + UNDERLINE + YELLOW
    UNDERLINE = "\u001b[7m"  # Titles, Headers + BOLD + YELLOW, implemented as a reverse of font color/background color
    # on some terminals (e.g., PyCharm)
    RESET = "\033[0m"

    def __new__(cls: _TerminalScheme) -> _TerminalScheme:
        """
        Force color scheme as singleton

        """
        if not hasattr(cls, "instance"):
            cls.instance = super(_TerminalScheme, cls).__new__(cls)
        return cls.instance

    def __str__(self):
        return "Scheme for CalSciPy terminal printing"

    @property
    def type(self) -> str:
        """
        Style for type hinting

        """
        return self.BLUE

    @property
    def emphasis(self) -> str:
        """
        Style for emphasis

        """
        return self.YELLOW

    @property
    def header(self) -> str:
        """
        Style for headers, titles and other things of utmost importance

        """
        return self.BOLD + self.UNDERLINE + self.YELLOW

    @staticmethod
    def __name__() -> str:
        return "Terminal Scheme"

    def __repr__(self):
        return "Scheme for CalSciPy terminal printing"

    def __call__(self, message: str, style: str) -> str:
        """
        Returns properly formatted string without setting global style if appending additional messages

        :param message: string to be formatted
        :param style: desired format (type, emphasis, header or class var)

        :return: formatted string
        """
        # I could probably just have this fail directly, but this is a bit more graceful.
        # It's more important that the message to the user is received than raising an exception because of style
        # matters.

        with PatternMatching(style, eq) as case:
            if any([case(style_) for style_ in dir(self) if "__" not in style_]):
                return "".join([getattr(self, style), message, self.RESET])
            else:
                return "".join([message, f"--requested terminal printing style {style} does not exist"])


# instance color scheme
COLORS = _ColorScheme()

# instance terminal scheme
TERM_SCHEME = _TerminalScheme()
