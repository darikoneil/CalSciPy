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
    return np.hstack(traces_as_tensor)
# TODO FINISH


@validate_numpy_type(required_dtype="object", pos=0)
def merge_factorized_tensors(factorized_traces: np.ndarray(dtype=object), component: int = 0) -> np.ndarray:
    """
    Concatenate a neuron x chunk or trial array in which each element is a component x frame factorization of the
    original trace:


    :param factorized_traces:
    :param component:
    :return:
    :rtype: np.ndarray
    """
    _neurons, _trials = factorized_traces.shape
    _frames = np.concatenate(factorized_traces[0], axis=1)[0, :].shape[0]
    traces_as_matrix = np.full((_neurons, _frames), 0, dtype=factorized_traces.dtype)  # pre-allocated

    # Merge Here - could be done more efficiently but not priority
    for _neuron in tqdm(
            range(_neurons),
            total=_neurons,
            desc="Generating Trace Matrix",
            disable=False,
    ):
        traces_as_matrix[_neuron, :] = np.concatenate(factorized_traces[_neuron], axis=1)[_component, :]

    return traces_as_matrix
# TODO TEST-ME
