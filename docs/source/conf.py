import os
import sys
import tomli
from datetime import date
from pathlib import Path


# IMPORTS ps I can be done not so dumbly
sys.path.insert(0, os.path.dirname(os.path.dirname(os. getcwd())))

# get package details directly from pyproject
pyproject_file = Path.cwd().parents[1].joinpath("pyproject.toml")
with open(pyproject_file, "rb") as file:
    package_details = tomli.load(file).get("project")

# get date for copyright
today = f"{date.today().year}"
authors = package_details.get("authors")[0].get("name")
author = f"{authors}"  # f-string because maybe weird sphinx stuff if it gets list, not sure
project = package_details.get("name")
copyright = "".join([today, author])
release = package_details.get("version")


extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx_autodoc_typehints']

typehints_defaults = "comma"

source_suffix = ".rst"

language = "en"

intersphinx_mapping = {
    'cupy': ('https://docs.cupy.dev/en/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/1.24/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'scikit-image': ('https://scikit-image.org/docs/stable/', None)
}

html_theme = 'sphinx_rtd_theme'

pygments_style = "sphinx"

latex_engine = "pdflatex"

typehints_use_signature = True

typehints_use_signature_return = True

todo_include_todos = True

autodoc_mock_import = ["win32com", "pywin32"]