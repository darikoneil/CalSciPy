@echo off

sphinx-apidoc -o docs\source CalSciPy -f -e

cd docs\source

DEL modules.rst

cd ..

cd ..
