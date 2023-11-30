from __future__ import annotations
from pathlib import Path
from .._user import verbose_copying
from .experiment import Experiment


class PrescreenExperiment(Experiment):
    def __init__(self, name: str, base_directory: Path, **kwargs):
        """
        Stimulation experiment mix-in

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
        self.file_tree.add_path("stimulation")
        super().generate_class_files()

    def export_protocol(self, directory: Path) -> int:
        stim_protocol_dir = self.file_tree.get("stimulation")()
        assert len([file for file in stim_protocol_dir.rglob("*") if file.is_file()]) >= 1, "Missing protocol files!"
        verbose_copying(self.file_tree.get("stimulation")(), directory, "stimulation protocols")
        return 0
