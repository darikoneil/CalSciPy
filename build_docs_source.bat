@echo off

sphinx-apidoc -o docs\source src\CalSciPy -f -e

cd docs\source

DEL modules.rst
