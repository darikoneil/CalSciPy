from __future__ import annotations
from typing import Tuple
from pathlib import Path
from xml.etree import ElementTree


import pandas as pd
import numpy as np

from .._validators import convert_permitted_types_to_required, validate_extension
from .experiment import Experiment
from .._user import select_directory, verbose_copying
from .files import FileSet


class ImagingExperiment(Experiment):
    def __init__(self, name: str, base_directory: Path, **kwargs):
        """
        Imaging Experiment mix-in

        :param name: name of experiment
        :type name: str
        :param base_directory: base directory of mouse
        :type base_directory: Path
        :key mix_ins: an iterable of mix-ins in string or object form
        """
        # noinspection PyArgumentList
        super().__init__(name, base_directory, **kwargs)

    def collect_data(self) -> Experiment:
        """
        Implementation of abstract method for data collection

        :rtype: Experiment
        """
        imaging_directory = \
            select_directory(title="Select folder containing raw imaging data")
        verbose_copying(imaging_directory, self.file_tree.get("imaging")(),
                        content_string="imaging data")
        self.reindex()
        super().collect_data()

    def analyze_data(self) -> Experiment:
        """
        Implementation of abstract method for analyzing data

        :rtype: Experiment
        """
        raise NotImplementedError

        super().analyze_data()

    def generate_class_files(self) -> Experiment:
        """
        Implementation of abstract method for generating file sets specific to mix-in

        :rtype: Experiment
        """
        self.file_tree.add_path("imaging")
        super().generate_class_files()
