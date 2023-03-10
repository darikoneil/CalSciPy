import pytest
import os


# Adjust base dir if necessary
base_dir = os.getcwd()
if "tests" not in base_dir:
    base_dir = os.path.join(base_dir, "tests")

# Turn off JIT for proper coverage
os.environ["NUMBA_DISABLE_JIT"] = "1"
