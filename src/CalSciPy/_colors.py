from __future__ import annotations
from typing import Tuple, Union


class _Colors:
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
    mapping = list(enumerate([red, green, blue, orange, purple, yellow, black, dark, medium, light, background, white]))

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(_Colors, cls).__new__(cls)
        return cls.instance

    def __call__(self, value: Union[int, str], *args, **kwargs):
        if isinstance(value, int):
            if value >= len(self.mapping):
                value = len(self.mapping) % value
            return self.mapping[value][1]
        elif isinstance(value, str):
            return getattr(self, value)


# instance
COLORS = _Colors()
