from __future__ import annotations
from typing import Any
import sys
from sys import float_info
from os import devnull
from shutil import copytree, rmtree
from pathlib import Path

import numpy as np


# Simple class that blocks printing
class BlockPrinting:
    """
    Simple class that blocks printing

    """
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = open(devnull, "w")

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any):
        sys.stdout.close()
        sys.stdout = self._stdout


# MARK TO USE NEW COPY OF DATASET EACH TIME SO ORIGINAL IS EFFECTIVELY IMMUTABLE
def copy_dummies(source_folder, dest_folder) -> None:
    """
    Copy dummy files to testing dir to avoid mutations concerns

    :param source_folder: source of dummy data
    :param dest_folder: destination of temp folder
    """
    copytree(source_folder, dest_folder, dirs_exist_ok=True)


def generate_dummy_file_name(name, dummy_folder, sub_folder):
    if sub_folder:
        dummy_folder = dummy_folder.joinpath(sub_folder)

    return dummy_folder.joinpath(name)


def generate_dummy_output_folder(name, dummy_folder):

    output_folder = dummy_folder.joinpath("".join(["output_", name]))

    if not output_folder.exists():
        output_folder.mkdir()

    return output_folder


def identify_dummy_source() -> Path:
    """
    Identifies dummy source folder

    """
    # Adjust base dir if necessary
    base_dir = Path.cwd()
    if "tests" not in str(base_dir):
        base_dir = base_dir.joinpath("tests")

    # Find directory containing testing / dummy samples
    return base_dir.joinpath("testing_samples")


def purge_dummies(dest_folder) -> None:
    """
    Purge dummy files in testing dir to avoid mutation concerns

    :param dest_folder: destination of temp folder
    """
    rmtree(dest_folder)


def read_descriptions(file):
    return np.genfromtxt(str(file), delimiter=",", dtype="int")

