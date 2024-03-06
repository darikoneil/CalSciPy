from __future__ import annotations
from typing import Iterable
from pathlib import Path
from abc import abstractmethod
from importlib import import_module

from .files import FileTree, FileSet
from ._logging_tools import get_timestamp


def import_mix_in_string(json_string: str) -> Experiment:
    """
    Import experiment mix-in from string. Usually this is taken from the serialized object but not always.
    For this to always work, we have to make sure that the mix-in follows PEP naming conventions and resides in its own
    module

    :param json_string: string indicating mix-in to import
    :type json_string: str
    :return: imported mix-in
    :rtype: Experiment

    .. versionadded:: 0.8.1

    .. warning:: Currently untested

    """
    module_name = [char_ if char_.islower() else "".join(["_", char_.lower()])
                   for char_ in json_string if char_.lower()]
    module_name = "".join(module_name)
    module_name = "".join(module_name[1:])
    return getattr(import_module("".join(["CalSciPy.organization.", module_name])), json_string)


class Experiment:
    def __init__(self, name: str, base_directory: Path, **kwargs) -> Experiment:
        """
        Abstract experiment class that collects its methods through mix-ins. These mix-ins add experimental features to
        the object. For example, the imaging mix-in adds methods for collecting imaging data for placement in the
        file tree

        :param name: name of experiment
        :type name: str
        :param base_directory: base directory of mouse
        :type base_directory: pathlib.Path
        :key mix_ins: an iterable of mix-ins in string or object form

        .. versionadded:: 0.8.1

        .. warning:: Currently untested

        """
        #: str: name of the experiment
        self._name = name
        #: Path: base directory of mouse
        self._base_directory = base_directory
        #: Iterable: iterable of mix-ins in string or object form
        self._mix_ins = kwargs.get("mix_ins", [])
        #: pd.DataFrame: synchronized experiment data
        self.data = None
        #: dict: file tree experimental folders and files
        self.file_tree = FileTree(self._name, base_directory)
        #: str: instance date
        self._instance_date = get_timestamp()

        self.generate_file_tree()

    @staticmethod
    def __name__() -> str:
        return "Experiment"

    @classmethod
    def __json_construct__(cls: object, self: object) -> Experiment:
        """
        Constructs the experiment from serialized form

        :rtype: Experiment
        """
        factory = ExperimentFactory(name=self._name, base_directory=self._base_directory)
        factory.add_mix_ins([import_mix_in_string(mix_in) for mix_in in self._mix_ins])
        experiment = factory.instance_constructor()
        for key, value in vars(self).items():
            if key != "_mix_ins":
                setattr(experiment, key, value)
            else:
                setattr(experiment, key, [import_mix_in_string(mix_in) for mix_in in self._mix_ins])
        return experiment

    def get(self, *args, **kwargs) -> FileSet:
        return self.file_tree.get(*args, **kwargs)

    def reindex(self) -> Experiment:
        """
        Updates file tree

        :rtype: Experiment
        """
        self.file_tree.reindex()

    def remap(self, base_directory: Path) -> Experiment:
        """
        Remaps file tree to a new base_directory, allowing us to move our folder without destroying our file tree.

        :param base_directory: base directory of mouse
        :type base_directory: pathlib.Path
        :rtype: Experiment
        """
        self._base_directory = base_directory
        self.file_tree.remap(base_directory)

    def validate(self) -> Experiment:
        self.file_tree.validate()

    @abstractmethod
    def collect_data(self) -> Experiment:
        """
        Abstract method for collecting experimental data and organizing into the file tree

        :rtype: Experiment
        """
        pass

    @abstractmethod
    def analyze_data(self) -> Experiment:
        """
        Abstract method for analyzing the data within the file tree

        :rtype: Experiment
        """
        pass

    @abstractmethod
    def generate_class_files(self) -> Experiment:
        """
        Abstract method for generating any file sets within the file tree that are specific to some mix-in

        :rtype: Experiment
        """
        pass

    def generate_file_tree(self) -> Experiment:
        """
        Method generates the experiment's file tree

        :rtype: Experiment
        """
        self.file_tree.add_path("results")
        self.file_tree.add_path("figures")
        self.generate_class_files()
        self.file_tree.build()

    def __json_encode__(self) -> dict:
        """
        Method encodes the object into a serializable dictionary

        :rtype: dict
        """
        serial_encoding = {key: (value if key != "_mix_ins" else [str(value_.__name__) for value_ in value])
                           for key, value in vars(self).items()}
        return serial_encoding


class ExperimentFactory:
    def __init__(self, name: str, base_directory: Path = None):
        """
        Factory for dynamically creating an experiment using the abstract experiment class and an iterable of mix-ins

        :param name: name of experiment
        :type name: str
        :param base_directory: base directory of mouse
        :type base_directory: pathlib.Path = None

        .. versionadded:: 0.8.1

        .. warning:: Currently untested

        """
        #: str: name of experiment
        self._name = name
        #: Path: base directory of mouse
        self.base_directory = base_directory
        #: Iterable: iterable of mix-ins in string or object form
        self._mix_ins = []

    def add_mix_ins(self, mix_ins: Iterable) -> ExperimentFactory:
        """
        Add mix-ins to the factory to include them when generating the experiment

        :param mix_ins: an iterable of mix-ins in string or object form
        :type mix_ins: Iterable
        :rtype: ExperimentFactory
        """
        for mix_in in mix_ins:
            if isinstance(mix_in, str):
                mix_in = import_mix_in_string(mix_in)
            self._mix_ins.append(mix_in)

    def object_constructor(self) -> Experiment:
        """
        Construct a concrete experiment object using the mix-ins

        :return: A concrete experiment object
        :rtype: Experiment
        """
        params = dict(self.__dict__)
        params.pop("base_directory")
        return type(self._name, tuple(self._mix_ins), params)

    def instance_constructor(self) -> Experiment:
        """
        Construct an instance of a concrete experiment object using the mix-ins

        :return: An instance of a concrete experiment object
        :rtype: Experiment
        """
        experiment_object = self.object_constructor()
        # noinspection PyCallingNonCallable
        return experiment_object(name=self._name, base_directory=self.base_directory, mix_ins=self._mix_ins)
