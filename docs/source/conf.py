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

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx_autodoc_typehints']

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

print(f"Current Path: {os.getcwd()}")
list_subfolders_with_paths = [f.path for f in os.scandir(os.getcwd()) if f.is_dir()]
print(f"Children: {list_subfolders_with_paths}")
test_path = "/home/docs/checkouts/readthedocs.org/user_builds/calscipy/checkouts/latest/src/CalSciPy/io.py"

parent = os.path.abspath('../..')
# print(f"Parent Path: {parent}")
next_path = "".join([str(parent), "\\src"])
sys.path.append("".join([str(parent), "\\src"]))
# print(f"Next Path: {next_path}")
next_path = "".join([str(parent), "\\src\\CalSciPy"])
# print(f"Next Path: {next_path}")
sys.path.append("".join([str(parent), "\\src\\CalSciPy"]))
next_path = "".join(["\\CalSciPy"])
# print(f"Next Path: {next_path}")
sys.path.append("".join(["\\CalSciPy"]))
# print(f"Next Path: {next_path}")
sys.path.append("".join([str(parent), "\\CalSciPy\\src"]))
# print(f"Next Path: {next_path}")
sys.path.append("".join([str(parent), "\\CalSciPy\\src\\CalSciPy"]))

for module in module_names:
    next_path = "".join([str(parent), "\\CalSciPy\\src\\CalSciPy\\", module])
    sys.path.append(next_path)
    # print(f"Next Path: {next_path}")
    next_path = "".join([str(parent), "\\src\\CalSciPy\\", module])
    sys.path.append(next_path)
    # print(f"Next Path: {next_path}")
    next_path = "".join([str(parent), "\\src\\CalSciPy\\", module, ".py"])
    sys.path.append(next_path)
