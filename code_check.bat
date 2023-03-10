@echo off
coverage run
coverage json
coverage lcov
coverage html
coverage report
flake8