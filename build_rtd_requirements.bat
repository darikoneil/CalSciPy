@echo off

pip freeze > docs\rtd_requirements.txt

cd docs

python -m truncate_requirements
