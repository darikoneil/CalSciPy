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

BETA STATUS/IN DEVELOPMENT/UNRELEASED/UNSTABLE/ETC:

ONLY PUBLIC TO LEVERAGE ACTIONS

This python packages contains a variety of useful methods for handling, processing, and visualizing calcium imaging data. It's intended to be a collection of useful, well-documented functions often used in boilerplate code alongside software packages such as [Caiman](https://github.com/flatironinstitute/CaImAn), [SIMA](https://github.com/losonczylab/sima), and [Suite2P](https://github.com/MouseLand/suite2p).

#### Highlights
* Assign unique colormaps to subsets of ROIs to generate rich, informative videos
* Perona-Malik diffusion for edge-preserving denoising
* Methods for handling Bruker's PrairieView data
* Interactive visualization

#### Installation
`pip install CalSciPy`

#### Subpackages
* Bruker - MAIN
* Coloring - TODO
* Event Processing - DEV
* Input/Output (I/O) - MAIN
* Image Processing - DEV
* Interactive Visuals - TODO
* Reorganization - MAIN
* Signal Processing - DEV
* Static Visuals - TODO

#### Documentation
Hosted at [ReadtheDocs](https://calscipy.readthedocs.io/en/latest/index.html#).
Available locally as [HTML](https://github.com/darikoneil/CalSciPy/tree/master/docs/build/html), [LATEX](https://github.com/darikoneil/CalSciPy/tree/master/docs/build/latex) and [PDF](https://github.com/darikoneil/CalSciPy/blob/master/docs/build/pdf/calscipy.pdf).

#### Roadmap
TODO - UNDOCUMENTED / OFFLINE     
DEV - UNTESTED / IN DEV BRANCH      
MAIN - COMPLETE /  IN MAIN BRANCH     

Generally completing each module before the next.     
Exceptions done last:       
* Bruker's Prairiview XML
* Bruker's Prairieview ENV parsing
* Trace Processing's Diffusion
* Event Processing's Covariance
* Interactive's ROI methods
