import pytest

from CalSciPy._user import verbose_copying


def test_copying(temp_path):
    # use suite2p folder because it has nested vars
    temp_folder = temp_path.joinpath("suite2p")
    comparison = temp_path.joinpath("copy_test_folder")

    verbose_copying(temp_folder, comparison)

    temp_files = sum([1 for file in temp_folder.rglob("*") if file.is_file()])
    comp_files = sum([1 for file in comparison.rglob("*") if file.is_file()])

    assert (temp_files == comp_files)
