from __future__ import annotations
from typing import Iterable, Union

import numpy as np
import pandas as pd

from ._validators import validate_evenly_divisible, validate_matrix, validate_tensor


"""
A few methods for converting/reorganizing data
"""


def align_data(data: pd.DataFrame,
               reference: pd.DataFrame,
               interpolate: str = None,
               join: bool = False,
               ) -> pd.DataFrame:
    """
    Aligns two datasets using their timestamps (index). The timestamps must be in the same units.

    :param data: Data to be aligned to the reference data

    :type data: :class:`DataFrame <pandas.DataFrame>`

    :param reference: Reference dataset. The function will align the data to its timestamps.

    :type reference: :class:`DataFrame <pandas.DataFrame>`

    :param interpolate: Whether to interpolate the missing values after alignment. Options include 'nearest',
        'slinear', 'quadratic', 'cubic', 'barycentric', 'krogh', 'piecewise_polynomial', 'pchip;, 'akima', and
        'cubicspline'

    :type interpolate: :class:`Optional <typing.Optional>`\[:class:`str`\], default: ``None``

    :param join: Whether to join the two datasets instead of returning only the aligned dataset

    :type join: :class:`bool`\, default: ``False``

    .. versionadded:: 0.8.0

    .. warning::
        The timestamps must be in the same units.

    """
    # we can't do in place in pandas
    data_ = data.copy(deep=True)

    # since reindex is strictly-label matching, check to make sure index types match or weird behavior could result
    try:
        assert (reference.index.dtype == data_.index.dtype)
    except AssertionError:
        # coerce type and try to reindex
        data_ = data_.reindex(index=data_.index.astype(reference.index.dtype))
        # make sure it worked and didn't just change the dtype to object
        assert (reference.index.dtype == data_.index.dtype,  # noqa: F631
                "Datasets must have timestamps with identical types and units")  # noqa: F631

    # reindex data
    data_ = data_.reindex(index=reference.index)

    # interpolate
    if interpolate is not None:
        data_.interpolate(method=interpolate, inplace=True)
        # forward fill the final frame
        data_.ffill(inplace=True)

    if join:
        # Join two datasets (deep copy to make sure not a view)
        reference_ = reference.copy(deep=True)
        return reference_.join(data_)
    else:
        return data_


def external_align(data: np.ndarray,
                   samples: pd.DataFrame,
                   reference: pd.DataFrame,
                   interpolate: bool = False,
                   join: bool = False,
                   tag: str = "Feature",
                   ) -> pd.DataFrame:
    """
    Aligns a numpy array with a reference dataset using the timestamped samples as an intermediary.

    :param data: Data to be aligned to the reference data

    :type data: :class:`ndarray <numpy.ndarray`

    :param samples: Timestamped samples. Defines the relationship between each data sample and the reference timestamps.

    :type samples: :class:`DataFrame <pandas.DataFrame>`

    :param reference: Reference dataset. The function will align the data to its timestamps.

    :type reference: :class:`DataFrame <pandas.DataFrame>`

    :param interpolate: Whether to interpolate the missing values after alignment. Options include 'nearest',
        'slinear', 'quadratic', 'cubic', 'barycentric', 'krogh', 'piecewise_polynomial', 'pchip;, 'akima', and
        'cubicspline'

    :type interpolate: :class:`Optional <typing.Optional>`\[:class:`str`\], default: ``None``

    :param join: Whether to join the two datasets instead of returning only the aligned dataset

    :type join: :class:`bool`\, default: ``False``

    :param tag: String prefix for describing each row in the data

    .. versionadded:: 0.8.0

    .. warning::
        The timestamps must be in the same units.
    """
    # Make sure samples and reference have identical timestamps
    try:
        assert (reference.index == samples.index)
        samples_ = samples.copy(deep=True)
    except AssertionError:
        samples_ = align_data(samples, reference)

    # calculate filled timestamps
    samples_ = samples_.dropna()

    # calculate first / last sample; not we assume here no missing samples between data timestamps and data
    first_sample = samples_.min()
    last_sample = samples_.max()
    inc_data = data[:, int(first_sample):int(last_sample) + 1]

    # generate dataframe
    column_names = ["".join([tag, f"{idx}"]) for idx in range(inc_data.shape[0])]
    timestamped_data = pd.DataFrame(data=inc_data.T, index=samples_.index, columns=column_names)

    # align
    return align_data(timestamped_data, reference=reference, interpolate=interpolate, join=join)


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
