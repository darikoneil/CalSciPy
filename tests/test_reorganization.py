import pytest
import numpy as np
import os

from PPVD.style import TerminalStyle
from CalSciPy.reorganization import generate_raster, generate_tensor, merge_factorized_matrices, merge_tensor


# arguments for tests
sample_event_lists = (
    [([[1, 2, 3], [5, 6, 7], [0, 10, 20]], None),
     ([[1, 2, 3], [5, 6, 7], [0, 10, 20]], 21)]
)


sample_factorized_matrix = np.load(os.path.join(
    os.getcwd(), "testing_data", "sample_variables", "sample_factorized_matrices.npy"
), allow_pickle=True)


sample_matrix = np.full((5, 100), 1)
sample_matrix[0, :] = np.arange(100)

sample_tensor = np.array([np.full((4, 5), 1), np.full((4, 5), 2), np.full((4, 5), 3)])


@pytest.mark.parametrize("matrix", [sample_matrix])
def test_generate_tensor_passes(matrix):
    tensor = generate_tensor(matrix, 25)
    if tensor.shape != (4, 5, 25):
        raise AssertionError(f"Generating tensor did not maintain correct shape: expected"
                             f" {(4, 5, 25)} received {tensor.shape}")
    for _chunk in range(tensor.shape[0]):
        np.testing.assert_array_equal(tensor[_chunk, 0, :], np.arange(25)+(_chunk*25), f"Chunk {_chunk}: Generating"
                                                                                       f" tensor did not maintain"
                                                                                       f" correct order")


def test_generate_tensor_fails():
    # validate not matrix
    with pytest.raises(AssertionError):
        generate_tensor(np.full((5, 90, 5), 1), 25)
    # validate not evenly divisible
    with pytest.raises(AssertionError):
        generate_tensor(np.full((5, 90), 1), 25)


@pytest.mark.parametrize(("event_frames", "total_frames"), [sample_event_lists[0], sample_event_lists[1]])
def test_generate_raster_passes(event_frames, total_frames):
    event_matrix = generate_raster(event_frames, total_frames)
    if event_matrix.shape[0] != 3:
        raise AssertionError(f"Generate Raster did not generate the correct number of neurons: input ="
                             f" {event_frames} given {total_frames} frames")

    if event_matrix.shape[1] != 21:
        raise AssertionError(f"Generate Raster did not generate the correct total frames: input = {event_frames} given"
                             f" {total_frames} frames")

    np.testing.assert_array_equal(np.sum(event_matrix, axis=0),
                                  np.array([1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
                                  err_msg=f"Generate Raster did not correctly place the events input = {event_frames}"
                                          f" given {total_frames} frames & returned {np.sum(event_matrix, axis=0)}")


@pytest.mark.parametrize("traces_as_tensor", [sample_tensor])
def test_merge_tensor_passes(traces_as_tensor):
    traces_as_matrix = merge_tensor(traces_as_tensor)
    if traces_as_matrix.shape != (4, 15):
        raise AssertionError(f"Merge Tensor did not maintain correct shape: expected {(4, 15)} received"
                             f" {traces_as_matrix.shape}")


def test_merge_tensor_fails():
    # validate tensor
    with pytest.raises(AssertionError):
        merge_tensor(np.full((5, 100), 1))


@pytest.mark.parametrize("factorized_matrix", [sample_factorized_matrix])
def test_merge_factorized_matrices_passes(factorized_matrix):

    for _component in range(3):
        component_matrix = merge_factorized_matrices(factorized_matrix, component=_component)
        if component_matrix.shape != (10, 450):
            raise AssertionError(f"Component {_component}: Merging factorized matrices did not maintain correct shape:"
                                 f" expected {(10, 450)} received {TerminalStyle.BLUE}{component_matrix.shape}")
        np.testing.assert_array_equal(component_matrix[_component, :], np.concatenate([np.arange(100)*_component,
                                                                                       np.arange(100)*_component,
                                                                                       np.arange(100)*_component,
                                                                                       np.arange(100) * _component,
                                                                                       np.arange(50)*_component]),
                                      f"Component {_component}: Merging factorized matrices did not maintain correct"
                                      f" order")


def test_merge_factorized_matrices_fails():
    with pytest.raises(TypeError):
        merge_factorized_matrices(np.full((5, 90), 1, dtype=np.float64))
