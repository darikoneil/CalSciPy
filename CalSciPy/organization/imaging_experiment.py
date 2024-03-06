from __future__ import annotations
from pathlib import Path

from .._user import select_directory, verbose_copying
from .experiment import Experiment


class ImagingExperiment(Experiment):
    def __init__(self, name: str, base_directory: Path, **kwargs):
        """
        Imaging Experiment mix-in

        :param name: name of experiment
        :type name: str
        :param base_directory: base directory of mouse
        :type base_directory: Path
        :key mix_ins: an iterable of mix-ins in string or object form

        .. versionadded:: 0.8.1

        .. warning:: Currently untested

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
        _ = verbose_copying(imaging_directory, self.file_tree.get("imaging")(), content_string="imaging data")
        self.reindex()
        super().collect_data()

    def analyze_data(self) -> Experiment:
        """
        Implementation of abstract method for analyzing data

        :rtype: Experiment
        """
        raise NotImplementedError
        # noinspection PyUnreachableCode
        super().analyze_data()

    def generate_class_files(self) -> Experiment:
        """
        Implementation of abstract method for generating file sets specific to mix-in

        :rtype: Experiment
        """
        self.file_tree.add_path("imaging")
        super().generate_class_files()
