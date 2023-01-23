from __future__ import annotations
import numpy as np
from ._validation import validate_tensor


@validate_tensor(pos=0)
def merge_traces(traces_as_tensor: np.ndarray) -> np.ndarray:
    """
    Concatenate multiple trials or tiffs into single matrix:


    :param traces_as_tensor:
    :return:
    :rtype: np.ndarray
    """
    _neurons, _trials = traces_as_tensor.shape
    _frames = np.concatenate(traces_as_tensor, axis=1)[0, :].shape[0]
    traces_as_matrix = np.full((_neurons, _frames), 0, dtype=traces_as_tensor.dtype)  # pre-allocated
    return traces_as_matrix
# TODO FINISH REFACTOR
