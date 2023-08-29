from __future__ import annotations
from typing import Tuple, Union, List
from operator import eq

from ._backports import PatternMatching


class ColorScheme:
    """
    A container class for CalSciPy's color scheme

    """
    BLUE: Tuple[float, float, float] = (15 / 255, 159 / 255, 255 / 255)
    ORANGE: Tuple[float, float, float] = (255 / 255, 159 / 255, 15 / 255)
    GREEN: Tuple[float, float, float] = (64 / 255, 204 / 255, 139 / 255)
    RED: Tuple[float, float, float] = (255 / 255, 78 / 255, 75 / 255)
    PURPLE: Tuple[float, float, float] = (120 / 255, 64 / 255, 204 / 255)
    YELLOW: Tuple[float, float, float] = (255 / 255, 240 / 255, 15 / 255)
    SHADOW: Tuple[float, float, float] = (224 / 255, 224 / 255, 224 / 255)
    LIGHT: Tuple[float, float, float] = (192 / 255, 192 / 255, 192 / 255)
    MEDIUM: Tuple[float, float, float] = (128 / 255, 128 / 255, 128 / 255)
    DARK: Tuple[float, float, float] = (65 / 255, 65 / 255, 65 / 255)
    BLACK: Tuple[float, float, float] = (0 / 255, 0 / 255, 0 / 255)
    WHITE: Tuple[float, float, float] = (255 / 255, 255 / 255, 255 / 255)
    BACKGROUND: Tuple[float, float, float] = (245 / 255, 245 / 255, 245 / 255)
    colors: Tuple[str, ...] = ("red", "green", "blue", "orange", "purple", "yellow", "black", "medium", "dark", "light")

    def __new__(cls: ColorScheme) -> ColorScheme:
        """
        Force color scheme as singleton

        """
        if not hasattr(cls, "instance"):
            cls.instance = super(ColorScheme, cls).__new__(cls)
        return cls.instance

    @property
    def mapping(self) -> List[Tuple[int, Tuple[float, float, float]]]:
        """
        Mapping of integers to ordered color scheme

        :rtype: :class:`List <typing.List>`\[:class:`Tuple <typing.Tuple>`\[:class:`int`\,
            :class:`Tuple <typing.Tuple>`\[:class:`float`\, :class:`float`\, :class:`float`\]]]
        """
        return list(enumerate(self.colors))

    @property
    def num_colors(self) -> int:
        """
        Total number of colors

        :rtype: :class:`int`
        """
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
        else:
            raise TypeError("color scheme accepts int or str args only")


class TerminalScheme:
    """
    A container class for CalSciPy's terminal printing color/font scheme

    """
    BLUE: str = "\u001b[38;5;39m"  # Types
    YELLOW: str = "\u001b[38;5;11m"  # Emphasis
    BOLD: str = "\u001b[1m"  # Titles, Headers + UNDERLINE + YELLOW
    UNDERLINE: str = "\u001b[7m"  # Titles, Headers + BOLD + YELLOW, implemented as a reverse of font color/background color
    # on some terminals (e.g., PyCharm)
    RESET: str = "\033[0m"

    def __new__(cls: TerminalScheme) -> TerminalScheme:
        """
        Force color scheme as singleton

        """
        if not hasattr(cls, "instance"):
            cls.instance = super(TerminalScheme, cls).__new__(cls)
        return cls.instance

    def __str__(self):
        return "Scheme for CalSciPy terminal printing"

    @property
    def type(self) -> str:
        """
        Style for type hinting

        :rtype: :class:`str`
        """
        return self.BLUE

    @property
    def emphasis(self) -> str:
        """
        Style for emphasis

        :rtype: :class:`str`
        """
        return self.YELLOW

    @property
    def header(self) -> str:
        """
        Style for headers, titles and other things of utmost importance

        :rtype: :class:`str`
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
COLORS = ColorScheme()

# instance terminal scheme
TERM_SCHEME = TerminalScheme()
