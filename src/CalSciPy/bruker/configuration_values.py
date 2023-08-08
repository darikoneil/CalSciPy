from configparser import ConfigParser
from pathlib import Path

"""Configuration Constants"""

_config_parser = ConfigParser()

_working_directory = Path.cwd()

_config_loc = _working_directory.joinpath("src").joinpath("CalSciPy").joinpath("bruker").joinpath("config.ini")

_config = _config_parser.read(str(_config_loc))

# These are just in case there are major breaking changes
DEFAULT_PRAIRIEVIEW_VERSION = _config_parser.get("PRAIRIEVIEW", "VERSION")

BRUKER_XML_OBJECT_MODULES = _config_parser.get("CALSCIPY", "BRUKER_XML_OBJECT_MODULES")
