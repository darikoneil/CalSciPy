import os
import sys
import pathlib

# IMPORTS ps I can be done not so dumbly

sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.path.abspath('../../'))

module_names = (
    "bruker",
    "coloring",
    "event_processing",
    "image_processing",
    "interactive_visuals",
    "io_tools",
    "reorganization"
    "static_visuals",
    "trace_processing"
)

parent = str(pathlib.Path(os.getcwd()).parents[1])
sys.path.append("".join([parent, "\\src\\CalSciPy"]))
sys.path.append("".join([parent, "\\src"]))
for _module in module_names:
    sys.path.append("".join([parent, "\\src\\CalSciPy\\", _module]))

project = 'CalSciPy'
# noinspection PyShadowingBuiltins
copyright = "2023, Darik A. O'Neil"
author = "Darik A. O'Neil"
release = CalSciPy.version


extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx_autodoc_typehints']

typehints_defaults = 'comma'
templates_path = ['_templates']
exclude_patterns = []

language = '[en]'

intersphinx_mapping = {
    'cupy': ('https://docs.cupy.dev/en/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'python': ('https://docs.python.org/3', None),
    'numba': ('https://numba.pydata.org/', None),
    'numpy': ('https://numpy.org/doc/1.24/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'scikit-image': ('https://scikit-image.org/docs/stable/', None)

}

html_theme = 'sphinx_rtd_theme'
