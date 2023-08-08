from configparser import ConfigParser


"""Configuration Constants"""

_config_parser = ConfigParser()

_config = _config_parser.read(
    "C:\\Users\\YUSTE\\PycharmProjects\\pyPrairieView\\src\\pyPrairieView\\pyprairieview_config.ini")

# These are just in case there are major breaking changes
DEFAULT_PRAIRIEVIEW_VERSION = _config_parser.get("PRAIRIEVIEW", "VERSION")

BRUKER_XML_OBJECT_MODULES = _config_parser.get("PYPRAIRIEVIEW", "BRUKER_XML_OBJECT_MODULES")
