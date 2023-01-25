import os
import sys
import pathlib

sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.path.abspath('../../'))
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'CalSciPy'
# noinspection PyShadowingBuiltins
copyright = "2023, Darik A. O'Neil"
author = "Darik A. O'Neil"
release = '0.0.7'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

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
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/1.24/', None),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
# html_static_path = ['_static']

# IMPORTS

module_names = (
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
