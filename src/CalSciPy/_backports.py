from __future__ import annotations
from typing import Iterable, Any


class PatternMatching:
    def __init__(self, value: Any, comparison_expressions: Iterable[Any]):
        """
        Manual implementation of pattern matching for python < 3.10

        Not the most extensible or robust right now, but she works well for the
        current implementations.


        :param value: value or iterator of values of interest
        """
        self.value = value
        self.comparison_expressions = self._coerce_iterable(comparison_expressions)

    @staticmethod
    def _coerce_iterable(comparison_expression: Any) -> Iterable[Any]:
        if isinstance(comparison_expression, Iterable):
            return comparison_expression
        else:
            return [comparison_expression, ]  # return as iterable

    def __enter__(self):
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any):
        return False

    def __call__(self, cases: Any):
        """
        Magic call for comparing some case with some value using some comparison operator

        :param cases: case/s for comparison
        :returns: whether the case/s are matched by the value/s
        """
        for value, comparator, case in zip(self.value, self.comparison_expressions, cases):
            if not comparator(value, case):
                return False
        return True
