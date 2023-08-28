from importlib_metadata import metadata as _metadata


"""
Simply some manual integration of metadata
"""


_meta = _metadata("CalSciPy")

#: str: CalSciPy
name = _meta["name"]
#: str: CalSciPy author
author = _meta["author"]
#: str: CalSciPy maintainer
maintainer = _meta["maintainer"]
#: str: CalSciPy version
version = _meta["version"]
