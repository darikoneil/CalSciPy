from __future__ import annotations
from typing import Any
import sys
from sys import float_info
from os import devnull


class BlockPrinting:
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = open(devnull, "w")

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any):
        sys.stdout.close()
        sys.stdout = self._stdout
