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
release = '0.0.4'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc']

templates_path = ['_templates']
exclude_patterns = []

language = '[en]'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# IMPORTS

module_names = (
    "coloring",
    "event_processing",
    "image_processing",
    "interactive_visuals",
    "io",
    "\\src\\CalSciPy\\reorganization",
    "src\\CalSciPy\\reorganization",
    "CalSciPy\\reorganization",
    "\\CalSciPy\\reorganization",
    "reorganization"
    "static_visuals",
    "trace_processing"
)

parent = os.path.abspath('../..')
sys.path.insert(0, parent)
sys.path.append("".join(["\\src"]))
sys.path.append("".join(["\\src\\CalSciPy"]))
sys.path.append("".join(["\\CalSciPy"]))

for _module in module_names:
    print("\n")
    _import_name_1 = "".join([str(parent), "\\src\\CalSciPy\\", _module])
    print(_import_name_1)
    print("\n")
    _import_name_2 = "".join([str(parent), "\\CalSciPy\\", _module])
    print(_import_name_2)
    print("\n")
    sys.path.append(_import_name_1)
    sys.path.append(_import_name_2)

# sys.path.append("C:\\Users\\Darik\\.conda\\envs\\CalSciPy\\src")
# sys.path.append("C:\\Users\\Darik\\.conda\\envs\\CalSciPy\\src\\CalSciPy")
# sys.path.append("C:\\Users\\Darik\\.conda\\envs\\CalSciPy\\src\\CalSciPy\\reorganization")

# for _name in module_names:
#    sys.path.append("".join(["C:\\Users\\Darik\\.conda\\envs\\CalSciPy\\src\\CalSciPy", _name]))
# Parent = pathlib.Path(os.getcwd()).parents[0]
# sys.path.insert(0, Parent.with_name("src"))
# sys.path.insert(0, "".join([str(Parent), "\\src\\CalSciPy"]))
# for _name in module_names:
#    sys.path.insert(0, "".join([os.path.abspath('../..'), "\\src\\CalSciPy\\", _name]))
