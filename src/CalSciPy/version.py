from importlib_metadata import metadata as _metadata

_meta = _metadata("CalSciPy")

name = _meta["name"]
author = _meta["author"]
maintainer = _meta["maintainer"]
version = _meta["version"]
