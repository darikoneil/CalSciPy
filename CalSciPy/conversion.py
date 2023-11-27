from __future__ import annotations
from typing import Iterable, Union

import numpy as np
import pandas as pd

from ._validators import validate_evenly_divisible, validate_matrix, validate_tensor


"""
A few methods for converting/reorganizing data
"""


def align_data(analog_data: pd.DataFrame,
               frame_times: pd.DataFrame,
               fill: bool = False,
               method: str = "nearest"
               ) -> pd.DataFrame:
    """
    Synchronizes analog data & imaging frames using the timestamp of each frame. Option to generate a second column
    in which the frame index is interpolated such that each analog sample matches with an associated frame.

    :param analog_data: analog data
    :param frame_times: frame timestamps
    :param fill: whether to include an interpolated column so each sample has an associated frame
    :param method: method for interpolating samples
    :returns: a dataframe containing time (index, ms) with aligned columns of voltage recordings/analog data and
        imaging frame

    .. versionadded:: 0.8.0
    """
    frame_times = frame_times.reindex(index=analog_data.index)

    # Join frames & analog (deep copy to make sure not a view)
    data = analog_data.copy(deep=True)
    data = data.join(frame_times)

    if fill:
        frame_times_filled = frame_times.copy(deep=True)
        frame_times_filled.columns = ["Imaging Frame (interpolated)"]
        frame_times_filled.interpolate(method=method, inplace=True)
        # forward fill the final frame
        frame_times_filled.ffill(inplace=True)
        data = data.join(frame_times_filled)

    return data


def interpolate_traces(traces, aligned_data, method="pchip") -> pd.DataFrame:
    """
    function

    :param traces: ok
    :param aligned_data: ok
    :param method: ko
    :return: ok

    .. versionadded:: 0.8.0
    """
    frame_times = aligned_data["Imaging Frame"].dropna().index
    inc_frames = frame_times.shape[0]
    true_frames = traces.shape[-1]
    frame_diff = inc_frames - true_frames

    if frame_diff == 0:
        inc_traces = traces
    elif frame_diff < 0:
        frame_range = (aligned_data.get("Imaging Frame").min(), aligned_data.get("Imaging Frame").max())
        inc_traces = traces[:, int(frame_range[0]):int(frame_range[-1])+1]
    else:
        raise AssertionError("Selected frames don't exist")

    traces = pd.DataFrame(data=inc_traces.T, index=frame_times, columns=[
        "".join(["Neuron ", str(x)]) for x in range(inc_traces.shape[0])
    ])
    traces = traces.reindex(aligned_data.index)
    traces.interpolate(method=method, inplace=True)
    return traces


@validate_matrix(pos=0, key="matrix")
@validate_evenly_divisible(numerator=0, denominator=1, axis=1)
def matrix_to_tensor(matrix: np.ndarray, chunk_size: int) -> np.ndarray:
    """
    Generates a tensor given chunk / trial indices

    :param matrix: Traces in matrix form (neurons x frames)

    :param chunk_size: Size of each chunk

    :returns: Traces as a tensor of trial x neurons x frames
    """
    return np.stack(np.hsplit(matrix, matrix.shape[1] // chunk_size), axis=0)


def merge_factorized_matrices(factorized_traces: np.ndarray, components: Union[int, Iterable[int]] = 0) -> np.ndarray:
    """
    Concatenate a neuron x chunk or trial array in which each element is a component x frame factorization of the
    original trace:


    :param factorized_traces: Neurons x chunks (trial, tif, etc) containing the neuron's trace factorized
        into several components

    :param components: Specific component to extract

    :returns: traces of specific Component in matrix form
    """
    if isinstance(components, Iterable):
        return np.dstack(
            [_merge_factorized_matrices(factorized_traces, component) for component in components]
        ).swapaxes(0, 2).swapaxes(1, 2)
    else:
        return _merge_factorized_matrices(factorized_traces, components)


@validate_tensor(pos=0, key="tensor")
def tensor_to_matrix(tensor: np.ndarray) -> np.ndarray:
    """
    Concatenate multiple trials or tiffs into single matrix:


    :param tensor: Chunk (trial, tif, etc) x neurons x frames

    :returns: Traces in matrix form (neurons x frames)
    """
    return np.hstack(tensor)


def _merge_factorized_matrices(factorized_traces: np.ndarray, component: int = 0) -> np.ndarray:
    """
    Concatenate a neuron x chunk or trial array in which each element is a component x frame factorization of the
    original trace:


    :param factorized_traces: Neurons x chunks (trial, tif, etc) containing the neuron's trace factorized
        into several components

    :param component: Specific component to extract

    :returns: Traces of specific component in matrix form
    """
    _neurons, _trials = factorized_traces.shape
    _frames = np.concatenate(factorized_traces[0], axis=1)[0, :].shape[0]
    traces_as_matrix = np.full((_neurons, _frames), 0, dtype=factorized_traces.dtype)  # pre-allocated

    # Merge Here - could be done more efficiently but not priority
    for _neuron in range(_neurons):
        traces_as_matrix[_neuron, :] = np.concatenate(factorized_traces[_neuron], axis=1)[component, :]

    return traces_as_matrix
