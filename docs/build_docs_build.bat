@echo off

cd docs

call build_clean

md build\html

md build\latex

md build\pdf

md build\log

call build_html

call build_latex

call build_pdf

cd ..
