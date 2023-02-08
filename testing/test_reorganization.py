import pytest
import numpy as np

from src.CalSciPy.style import TerminalStyle
from src.CalSciPy.reorganization import generate_raster, merge_tensor, merge_factorized_tensors

sample_event_lists = (
    [([[1, 2, 3], [5, 6, 7], [0, 10, 20]], None),
     ([[1, 2, 3], [5, 6, 7], [0, 10, 20]], 21)]
)

sample_tensor = np.array([np.full((4, 5), 1), np.full((4, 5), 2), np.full((4, 5), 3)])


@pytest.mark.parametrize(("event_frames", "total_frames"), [sample_event_lists[0], sample_event_lists[1]])
def test_generate_raster_passes(event_frames, total_frames):
    event_matrix = generate_raster(event_frames, total_frames)
    if event_matrix.shape[0] != 3:
        raise AssertionError(f"{TerminalStyle.YELLOW}Generate Raster did not generate the correct number of neurons: "
                             f"Input = {TerminalStyle.BLUE}{event_frames} {TerminalStyle.YELLOW}given"
                             f"{TerminalStyle.BLUE}{total_frames}{TerminalStyle.YELLOW} frames{TerminalStyle.RESET}")

    if event_matrix.shape[1] != 21:
        raise AssertionError(f"{TerminalStyle.YELLOW}Generate Raster did not generate the correct total frames: "
                             f"Input = {TerminalStyle.BLUE}{event_frames} {TerminalStyle.YELLOW}given"
                             f"{TerminalStyle.BLUE}{total_frames}{TerminalStyle.YELLOW} frames{TerminalStyle.RESET}")

    np.testing.assert_array_equal(np.sum(event_matrix, axis=0),
                                  np.array([1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
                                  err_msg=f"{TerminalStyle.YELLOW}Generate Raster did not correctly place the events"
                             f"Input = {TerminalStyle.BLUE}{event_frames} {TerminalStyle.YELLOW}given"
                             f"{TerminalStyle.BLUE}{total_frames}{TerminalStyle.YELLOW} frames and returned"
                                          f"{TerminalStyle.BLUE}{np.sum(event_matrix, axis=0)}{TerminalStyle.YELLOW}"
                                          f"instead of {TerminalStyle.BLUE}"
                                          f"{np.array([1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])}"
                                          f"{TerminalStyle.RESET}")


@pytest.mark.parametrize("traces_as_tensor", [sample_tensor])
def test_merge_tensor_passes(traces_as_tensor):
    traces_as_matrix = merge_tensor(traces_as_tensor)
    if traces_as_matrix.shape != (4, 15):
        raise AssertionError(f"{TerminalStyle.YELLOW}Merge Tensor did not maintain correct shape:"
                             f"expected {(4, 15)} received {TerminalStyle.BLUE}{traces_as_matrix.shape}"
                             f"{TerminalStyle.RESET}")
    return


def test_merge_factorized_tensors_passes():
    return
