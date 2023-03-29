import pytest
import numpy as np
from CalSciPy.reorganization import generate_raster, generate_tensor, merge_factorized_matrices, merge_tensor
from pytest_lazyfixture import lazy_fixture


@pytest.mark.parametrize(("sample", "expected"), [
    (lazy_fixture("spike_times"), True),
    (lazy_fixture("factorized_matrix"), False),
])
def test_generate_raster(sample, expected):
    if expected:
        sub_sample_0, sub_sample_1 = sample
        event_matrix = generate_raster(sub_sample_0)
        if event_matrix.shape[0] != 3:
            raise AssertionError(f"Generate Raster did not generate the correct number of neurons: input ="
                                 f" {sub_sample_0} given {21} frames")
        event_matrix = generate_raster(sub_sample_1, 21)
        if event_matrix.shape[1] != 21:
            raise AssertionError(f"Generate Raster did not generate the correct total frames: input = {sub_sample_1} "
                                 f"given {21} frames")

        np.testing.assert_array_equal(np.sum(event_matrix, axis=0),
                                      np.array([1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
                                      err_msg=f"Generate Raster did not correctly place the events input = "
                                              f"{sub_sample_1} given {21} frames & returned"
                                              f" {np.sum(event_matrix, axis=0)}")

    else:
        with pytest.raises(ValueError):
            generate_raster(sample, None)


@pytest.mark.parametrize(("sample", "expected"), [
    (lazy_fixture("matrix"), True),
    (lazy_fixture("factorized_matrix"), False),
    (lazy_fixture("tensor"), False),
    (lazy_fixture("spike_times"), False)
])
def test_generate_tensor(sample, expected):
    if expected:
        tensor = generate_tensor(sample, 25)
        if tensor.shape != (4, 5, 25):
            raise AssertionError(f"Generating tensor did not maintain correct shape: expected"
                                 f" {(4, 5, 25)} received {tensor.shape}")
        for _chunk in range(tensor.shape[0]):
            np.testing.assert_array_equal(tensor[_chunk, 0, :], np.arange(25)+(_chunk*25), f"Chunk {_chunk}: Generating"
                                                                                           f" tensor did not maintain"
                                                                                           f" correct order")
        # make sure fails when not evenly divisible
        with pytest.raises(AssertionError):
            generate_tensor(sample, 24)
    else:
        with pytest.raises((AssertionError, AttributeError, IndexError)):
            generate_tensor(sample)


@pytest.mark.parametrize(("sample", "expected"), [
    (lazy_fixture("tensor"), True),
    (lazy_fixture("factorized_matrix"), False),
    (lazy_fixture("matrix"), False),
    (lazy_fixture("spike_times"), False)
])
def test_merge_tensor(sample, expected):
    if expected:
        traces_as_matrix = merge_tensor(sample)
        if traces_as_matrix.shape != (4, 15):
            raise AssertionError(f"Merge Tensor did not maintain correct shape: expected {(4, 15)} received"
                                 f" {traces_as_matrix.shape}")
    else:
        with pytest.raises((AssertionError, AttributeError)):
            merge_tensor(sample)


@pytest.mark.parametrize(("sample", "expected"), [
    (lazy_fixture("factorized_matrix"), True),
    (lazy_fixture("tensor"), False),
    (lazy_fixture("matrix"), False),
    (lazy_fixture("spike_times"), False)
])
def test_merge_factorized_matrices(sample, expected):

    if expected:
        for _component in range(3):
            component_matrix = merge_factorized_matrices(sample, component=_component)
            if component_matrix.shape != (10, 450):
                raise AssertionError(f"Component {_component}:"
                                     f" Merging factorized matrices did not maintain correct shape:"
                                     f" expected {(10, 450)} received {component_matrix.shape}")
            np.testing.assert_array_equal(component_matrix[_component, :], np.concatenate([np.arange(100)*_component,
                                                                                           np.arange(100)*_component,
                                                                                           np.arange(100)*_component,
                                                                                           np.arange(100) * _component,
                                                                                           np.arange(50)*_component]),
                                          f"Component {_component}:"
                                          f" Merging factorized matrices did not maintain correct"
                                          f" order")
    else:
        with pytest.raises((TypeError, AttributeError)):
            merge_factorized_matrices(sample)
