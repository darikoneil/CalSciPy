import pytest
import numpy as np
from CalSciPy.misc import generate_blocks


@pytest.mark.parametrize("block_size", [0, 1, 2, 5, 15, 25])
@pytest.mark.parametrize("block_buffer", [0, 1, 2, 5, 15, 25])
def test_generate_blocks(long_tensor, block_size, block_buffer):

    # normally we pass frame index so we can calculate in place
    tensor_frames = list(range(long_tensor.shape[0]))

    # make sure we catch these edge cases
    if block_size <= 1:
        with pytest.raises(ValueError):
            blocks = generate_blocks(tensor_frames, block_size, block_buffer)
            next(blocks)
    elif block_buffer >= block_size:
        with pytest.raises(AssertionError):
            blocks = generate_blocks(tensor_frames, block_size, block_buffer)
            next(blocks)
    elif block_size >= len(tensor_frames):
        with pytest.raises(AssertionError):
            blocks = generate_blocks(tensor_frames, block_size, block_buffer)
            next(blocks)
    else:
        # actual tests
        blocks = generate_blocks(tensor_frames, block_size, block_buffer)
        blocked_data = []
        try:
            for block in blocks:
                blocked_data.append(long_tensor[block, :, :])
        except RuntimeError:
            pass

        if block_buffer == 0:
            np.testing.assert_equal(np.concatenate(blocked_data, axis=0), long_tensor, err_msg="Tensor mutated")

        if block_buffer > 0:
            for idx in range(len(blocked_data) - 1):
                np.testing.assert_equal(blocked_data[idx][-block_buffer:], blocked_data[idx + 1][:block_buffer],
                                        err_msg="Buffer Mismatch")
