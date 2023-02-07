from __future__ import annotations
from typing import List
import numpy as np
from ._validation import validate_tensor


def generate_raster(spike_times: List[List[int]], total_frames: int):
    """
    Generate raster from lists of spike times

    :param spike_times: list of spike times
    :type spike_times: List[List[int]]
    :param total_frames: total number of frames
    :type total_frames: int
    :return: spike matrix
    :rtype: np.ndarray
    """
    _neurons = spike_times.shape[0]
    spike_matrix = np.full((_num_neurons, total_frames), 0, dtype=np.int32)

    # Merge Here - could be done more efficiently but not priority
    for _neuron in range(_neurons):
        for _spike in spike_times[_neuron]:
            spike_matrix[_neuron, _spike] = 1
    return spike_matrix
# TODO TEST-ME


@validate_tensor(pos=0)
def merge_tensor(traces_as_tensor: np.ndarray) -> np.ndarray:
    """
    Concatenate multiple trials or tiffs into single matrix:


    :param traces_as_tensor: neurons x frames x chunk (trial, tiff, etc)
    :type traces_as_tensor: np.ndarray
    :return: traces in matrix form
    :rtype: np.ndarray
    """
    return np.hstack(traces_as_tensor)
# TODO TEST-ME


@validate_numpy_type(required_dtype="object", pos=0)
def merge_factorized_tensors(factorized_traces: np.ndarray(dtype=object), component: int = 0) -> np.ndarray:
    """
    Concatenate a neuron x chunk or trial array in which each element is a component x frame factorization of the
    original trace:


    :param factorized_traces: neurons x chunks (trial, tiff, etc) containing the neuron's trace factorized
    into several components
    :type factorized_traces: np.ndarray
    :param component: specific component to extract
    :return: traces of specific component in matrix form
    :rtype: np.ndarray
    """
    _neurons, _trials = factorized_traces.shape
    _frames = np.concatenate(factorized_traces[0], axis=1)[0, :].shape[0]
    traces_as_matrix = np.full((_neurons, _frames), 0, dtype=factorized_traces.dtype)  # pre-allocated

    # Merge Here - could be done more efficiently but not priority
    for _neuron in range(_neurons):
        traces_as_matrix[_neuron, :] = np.concatenate(factorized_traces[_neuron], axis=1)[_component, :]

    return traces_as_matrix
# TODO TEST-ME
