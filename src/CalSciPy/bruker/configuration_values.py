from configparser import ConfigParser
from pathlib import Path

"""Configuration Constants"""

_config_parser = ConfigParser()

_path = Path(__file__).with_name("config.ini")

_config = _config_parser.read(_path)

# These are just in case there are major breaking changes
DEFAULT_PRAIRIEVIEW_VERSION = _config_parser.get("PRAIRIEVIEW", "VERSION")

BRUKER_XML_OBJECT_MODULES = _config_parser.get("CALSCIPY", "BRUKER_XML_OBJECT_MODULES")
