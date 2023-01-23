import os
import sys
import pathlib

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
release = '0.0.5'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.autosectionlabel']

templates_path = ['_templates']
exclude_patterns = []

language = '[en]'

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
    "io",
    "reorganization"
    "static_visuals",
    "trace_processing"
)

parent = os.path.abspath('../..')
print(f"Parent Path: {parent}")
next_path = "".join([str(parent), "\\src"])
sys.path.append("".join([str(parent), "\\src"]))
print(f"Next Path: {next_path}")
next_path = "".join([str(parent), "\\src\\CalSciPy"])
print(f"Next Path: {next_path}")
sys.path.append("".join([str(parent), "\\src\\CalSciPy"]))
next_path = "".join(["\\CalSciPy"])
print(f"Next Path: {next_path}")
sys.path.append("".join(["\\CalSciPy"]))
print(f"Next Path: {next_path}")
sys.path.append("".join([str(parent), "\\CalSciPy\\src"]))
print(f"Next Path: {next_path}")
sys.path.append("".join([str(parent), "\\CalSciPy\\src\\CalSciPy"]))

for module in module_names:
    next_path = "".join([str(parent), "\\CalSciPy\\src\\CalSciPy\\", module])
    print(f"Next Path: {next_path}")
    next_path = "".join([str(parent), "\\src\\CalSciPy\\", module])
    print(f"Next Path: {next_path}")
    
