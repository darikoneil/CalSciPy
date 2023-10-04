from __future__ import annotations
from typing import Any, Iterable, Tuple
from pathlib import Path

from PPVD.style import TerminalStyle
from json_tricks import load, dump

from .logging_tools import IPythonLogger, ModificationLogger, get_timestamp
from .experiment import Experiment, ExperimentFactory
from .user_interaction import select_directory


DEFAULT_MOUSE = "default_mouse"
DEFAULT_PATH = Path.cwd()


class Mouse:

    #: list: modifications to this object
    _modifications = ModificationLogger()

    def __init__(self, name: str = DEFAULT_MOUSE, directory: Path = DEFAULT_PATH, study: str = None,
                 condition: str = None):
        """
        Class for organizing experiments for a single mouse

        :param name: name of mouse
        :type name: str
        :param directory: directory to save mouse within
        :type directory: pathlib.Path
        :param study: name of study
        :type study: str
        :param condition: experimental condition
        :type condition: str
        """

        # if directory doesn't contain mouse, we ought to add it
        if name not in directory.name:
            directory = directory.joinpath(name)

        # if directory doesn't exist, create it
        if not directory.exists():
            Path.mkdir(directory)

        #: IPython_logger: logging object
        self._logger = IPythonLogger(directory)

        #: str: directory to save mouse within
        self.directory = directory
        #: str: mouse
        self.name = name
        #: str: name of study
        self.study = study
        #: str: condition
        self.condition = condition
        #: str: instance date
        self._instance_date = get_timestamp()
        # call this only after all attrs successfully initialized
        self._modifications.append("Instantiated")

    def __str__(self) -> str:
        """
        Modified dunder method to print mouse attributes in a human-digestible form

        :rtype: str
        """
        attr_to_print = [("Mouse: ", self.name),
                         ("Instantiated: ", self._instance_date),
                         ("Study: ", self.study),
                         ("Condition: ", self.condition),
                         ("Experiments: ", self.experiments)
                         ]

        string_to_print = "\n"
        for attr in attr_to_print:
            key, value = attr
            if key == "Experiments: ":
                string_to_print += f"\n{TerminalStyle.BOLD}{TerminalStyle.YELLOW}{key}{TerminalStyle.RESET}\n"
                for experiment in value:
                    string_to_print += f"\t{experiment}\n"
            else:
                string_to_print += f"\n{TerminalStyle.BOLD}{TerminalStyle.YELLOW}{key}{TerminalStyle.RESET}{value}"

        string_to_print += f"\n\nLast modified: {TerminalStyle.GREEN}{self.modifications[0][0]}{TerminalStyle.RESET}" \
                           f", {TerminalStyle.GREEN}{self.modifications[0][1]}{TerminalStyle.RESET}"

        return string_to_print

    def save(self) -> Mouse:
        """
        Saves mouse to file

        :rtype: Mouse
        """
        # temporarily close logging
        self._logger.end_log()

        # dump is manipulative so:
        with open(self.organization_file, "w") as file:
            dump(self, file, indent=4)

        self._logger.start_log()

    @property
    def modifications(self) -> Tuple:
        return tuple(self._modifications)

    @property
    def experiments(self) -> Tuple:
        return tuple([name for name, experiment in vars(self).items() if isinstance(experiment, Experiment)])

    @property
    def organization_file(self) -> Tuple:
        return self.directory.joinpath("organization_file.json")

    @classmethod
    def load(cls: Mouse, directory: str = None) -> Mouse:
        """
        Function that loads a mouse

        :param directory: Directory containing the organization file, log file and associated data
        :type directory: str = None
        :return: mouse class instance
        :rtype: Mouse
        """
        if not directory:
            directory = select_directory(title="Select folder containing previously saved mouse", mustexist=True)

        organization_file = directory.joinpath("organization_file.json")

        with open(organization_file, "r") as file:
            mouse = load(file, preserve_order=False)

        # now we have to manually update our experiment mix-ins
        for key, value in vars(mouse).items():
            if isinstance(value, Experiment):
                experiment = getattr(mouse, key)
                setattr(mouse, key, experiment.__json_construct__(experiment))

        # update directory if we've moved our folder since then
        if mouse.directory != directory:
            mouse.directory = directory  # we don't need this check really,
            # but it's here atm to eventually add child updates

        return mouse

    def create_experiment(self, name: str, mix_ins: Iterable[Experiment]) -> Mouse:
        factory = ExperimentFactory(name=name, base_directory=self.directory)
        factory.add_mix_ins(mix_ins=mix_ins)
        setattr(self, name, factory.instance_constructor())

    def record(self, info: str = None) -> Mouse:
        """
        Record some modification

        :param info: a string containing a description of any changes
        :type info: str
        :rtype: Mouse
        """
        self._modifications.appendleft(info)

    def reindex(self) -> Mouse:
        """
        Updates dictionary for any all experiments

        :rtype: Mouse
        """
        for experiment_name in self.experiments:
            experiment = getattr(self, experiment_name)
            experiment.reindex()

    def validate(self) -> Mouse:
        """
        Validates all experimental files are accounted for

        :rtype: Mouse
        """
        for experiment_name in self.experiments:
            experiment = getattr(self, experiment_name)
            experiment.validate()

    def log_status(self) -> Mouse:
        return self._logger.check_log_status()

    def __json_encode__(self):
        serialized_mouse = {key: value for key, value in vars(self).items()}  # noqa: C416
        # unnecessary dict comprehension but now I can have big line saying make sure I'm a copy not a view

        # noinspection PyProtectedMember
        serialized_mouse["_modifications"] = self._modifications
        for key, value in serialized_mouse.items():
            if isinstance(value, Experiment):
                serialized_mouse[key] = {
                    "__instance_type__": ["mfr.experiment", "Experiment"],
                    "attributes": serialized_mouse[key].__json_encode__()
                }
        return serialized_mouse

    def __setattr__(self, key: Any, value: Any) -> Mouse:
        """
        Override magic to auto-record modifications

        :param key: key of attr being set
        :type key: Any
        :param value: value of attr being set
        :type value: Any
        :rtype: Mouse
        """
        super().__setattr__(key, value)
        self.record(key)

    def __del__(self):
        if "_logger" in vars(self):
            self._logger.end_log()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """
        Shuts down logging on exit

        :rtype: None
        """
        if "_logger" in vars(self):
            self._logger.end_log()
