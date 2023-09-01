from __future__ import annotations
from typing import Iterable, Union
import numpy as np

from ._validators import validate_evenly_divisible, validate_matrix, validate_tensor


"""
A few methods for converting/reorganizing data
"""


@validate_matrix(pos=0)
@validate_evenly_divisible(numerator=0, denominator=1, axis=1)
def matrix_to_tensor(traces_as_matrix: np.ndarray, chunk_size: int) -> np.ndarray:
    """
    Generates a tensor given chunk / trial indices

    :param traces_as_matrix: traces in matrix form (neurons x frames)

    :param chunk_size: size of each chunk

    :returns: traces as a tensor of trial x neurons x frames
    """
    return np.stack(np.hsplit(traces_as_matrix, traces_as_matrix.shape[1] // chunk_size), axis=0)


def merge_factorized_matrices(factorized_traces: np.ndarray, components: Union[int, Iterable[int]] = 0) -> np.ndarray:
    """
    Concatenate a neuron x chunk or trial array in which each element is a component x frame factorization of the
    original trace:


    :param factorized_traces: neurons x chunks (trial, tif, etc) containing the neuron's trace factorized
        into several components

    :param components: specific component to extract

    :returns: traces of specific component in matrix form
    """
    if isinstance(components, Iterable):
        return np.dstack(
            [_merge_factorized_matrices(factorized_traces, component) for component in components]
        ).swapaxes(0, 2).swapaxes(1, 2)
    else:
        return _merge_factorized_matrices(factorized_traces, components)


@validate_tensor(pos=0)
def tensor_to_matrix(traces_as_tensor: np.ndarray) -> np.ndarray:
    """
    Concatenate multiple trials or tiffs into single matrix:


    :param traces_as_tensor: chunk (trial, tif, etc) x neurons x frames

    :returns: traces in matrix form (neurons x frames)
    """
    return np.hstack(traces_as_tensor)


def _merge_factorized_matrices(factorized_traces: np.ndarray, component: int = 0) -> np.ndarray:
    """
    Concatenate a neuron x chunk or trial array in which each element is a component x frame factorization of the
    original trace:


    :param factorized_traces: neurons x chunks (trial, tif, etc) containing the neuron's trace factorized
        into several components

    :param component: specific component to extract

    :returns: traces of specific component in matrix form
    """
    _neurons, _trials = factorized_traces.shape
    _frames = np.concatenate(factorized_traces[0], axis=1)[0, :].shape[0]
    traces_as_matrix = np.full((_neurons, _frames), 0, dtype=factorized_traces.dtype)  # pre-allocated

    # Merge Here - could be done more efficiently but not priority
    for _neuron in range(_neurons):
        traces_as_matrix[_neuron, :] = np.concatenate(factorized_traces[_neuron], axis=1)[component, :]

    return traces_as_matrix
