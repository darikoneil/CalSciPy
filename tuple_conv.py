from __future__ import annotations
from typing import Tuple


def parse_tuple(vars, nested_type: typle) -> Tuple:

    parse_vars = []

    for var in vars.split(","):

        var = var.replace(" ", "")

        if "-" in var:
            start_val, end_val = var.split("-")
            start_val = int(start_val)
            end_val = int(end_val)
            var = list(range(start_val, end_val + 1, 1))
            var = str(var).replace("[", "").replace("]", "").replace(" ", "")

        parse_vars.extend([char for char in var.split(",")])

        return tuple([nested_type(var) for var in parse_vars])
