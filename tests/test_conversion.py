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
