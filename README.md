## CalSciPy       
<!-- Line 1 Badges... PyPi, Downloads, Maintained, Coverage, Documentation -->
<!-- Line 2 Badges... Python Versions, PyPi Status, License, Contributors -->
![PyPI](https://img.shields.io/pypi/v/CalSciPy)
![PyPI - Downloads](https://img.shields.io/pypi/dm/CalSciPy)
![Maintenance](https://img.shields.io/maintenance/yes/2023)
[![Coverage Status](https://coveralls.io/repos/github/darikoneil/CalSciPy/badge.svg?branch=master)](https://coveralls.io/github/darikoneil/CalSciPy?branch=master)
[![Documentation Status](https://readthedocs.org/projects/calscipy/badge/?version=latest)](https://calscipy.readthedocs.io/en/latest/?badge=latest)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/CalSciPy?)
![PyPI - Status](https://img.shields.io/pypi/status/CalSciPy)
![GitHub](https://img.shields.io/github/license/darikoneil/CalSciPy)
[![Contributors](https://img.shields.io/github/contributors-anon/darikoneil/CalSciPy)](https://github.com/darikoneil/CalSciPy/graphs/contributors)
![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/darikoneil/CalSciPy/calscipy_lint_test_action.yml)

This python packages contains a variety of useful tools for handling, processing, visualizing, & designing calcium 
imaging experiments. I noticed I was constantly re-writing boilerplate, "one-liners", and copy/pasting code between
environments. I wrote this so ~I don't have to~ you don't have to. It's intended to be a collection of useful, 
well-documented functions often used in boilerplate code alongside software packages such as 
[Caiman](https://github.com/flatironinstitute/CaImAn), [SIMA](https://github.com/losonczylab/sima), 
and [Suite2P](https://github.com/MouseLand/suite2p), as well as a collection of more bespoke software designed for use 
with Bruker's PrairieView Imaging Software and MeadowLark Spatial Light Modulators. Essentially, it's a collection of 
code written for my imaging experiments that might be considered useful to others.

#### Active Development
* The current implementation is unstable and should be considered an open beta.

#### Highlights
* Assign unique colormaps to subsets of ROIs to generate rich, informative videos
* Perona-Malik diffusion for edge-preserving denoising
* Fast, convenient functions for handling, aligning, and compiling data collected in Bruker's PrairieView software
* Functions for generating protocols for use in Bruker's Prairieview.
* Interactive visualization of data

#### Installation
`pip install CalSciPy`

#### Documentation
Hosted at [ReadtheDocs](https://calscipy.readthedocs.io/en/latest/index.html#).
Available locally as [HTML](https://github.com/darikoneil/CalSciPy/tree/master/docs/build/html), [LATEX](https://github.com/darikoneil/CalSciPy/tree/master/docs/build/latex) and [PDF](https://github.com/darikoneil/CalSciPy/blob/master/docs/build/pdf/calscipy.pdf).
