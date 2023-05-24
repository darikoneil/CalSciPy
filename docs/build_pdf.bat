@echo off

pdflatex -file-line-error -interaction=nonstopmode -synctex=1 -output-format=pdf -output-directory=build/pdf -aux-directory=build/log -include-directory=C:/Users/YUSTE/PycharmProjects/CalSciPy/src -include-directory=C:/Users/YUSTE/PycharmProjects/CalSciPy/docs/build/latex build/latex/calscipy.tex