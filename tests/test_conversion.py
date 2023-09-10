import pytest

import numpy as np

from CalSciPy.conversion import matrix_to_tensor, tensor_to_matrix, merge_factorized_matrices


def test_matrix_to_tensor(sample_matrix, sample_tensor):
    # test
    calc_tensor = matrix_to_tensor(sample_matrix, 10)
    np.testing.assert_array_equal(calc_tensor, sample_tensor)
    # ensure fails when not evenly divisible
    with pytest.raises(AssertionError):
        matrix_to_tensor(sample_matrix, 24)


def test_tensor_to_matrix(sample_tensor, sample_matrix):
    # test
    calc_matrix = tensor_to_matrix(sample_tensor)
    np.testing.assert_array_equal(calc_matrix, sample_matrix)


def test_factorized_matrix(sample_factorized_matrix):
    # single_component
    for component_ in range(3):
        component_matrix = merge_factorized_matrices(sample_factorized_matrix, components=component_)
        np.testing.assert_array_equal(component_matrix[component_, :], np.concatenate([np.arange(100) * component_,
                                                                                       np.arange(100) * component_,
                                                                                       np.arange(100) * component_,
                                                                                       np.arange(100) * component_,
                                                                                       np.arange(50) * component_])
                                      )
    # multi component
    components = merge_factorized_matrices(sample_factorized_matrix, components=[0, 2])
    comps = [merge_factorized_matrices(sample_factorized_matrix, components=c) for c in [0, 2]]
    for comparison in range(2):
        np.testing.assert_array_equal(components[comparison, :, :], comps[comparison])
