![Alt text](/docs/images/dancing_neuron_band.png?raw=true)
## Notice - 05/19/2024
I plan on breaking up this repo because its many of its contents are way too decoupled (e.g., bruker prairieview tricks / helpers vs experiment organization). All code related to computer-generated holography and holographic optogenetics will be migrated to my forthcoming neurobeam package (**N**euroscience **E**xperiments **U**sing **R**eal-time **O**ptogenetics, **BE**havior, and **M**icroscopy). All prairieview specific code will become a submodule. Experiment organization will become a submodule and contain functionality to convert to and from the NWB format. It will be some time before this is updated; neurobeam is my primary focus.

## CalSciPy      
<!-- Line 1 Badges... PyPi, Downloads, Maintained, Coverage, Documentation -->
<!-- Line 2 Badges... Python Versions, PyPi Status, License, Contributors -->
![PyPI](https://img.shields.io/pypi/v/CalSciPy)
![Maintenance](https://img.shields.io/maintenance/yes/2023)
[![Coverage Status](https://coveralls.io/repos/github/darikoneil/CalSciPy/badge.svg?branch=master)](https://coveralls.io/github/darikoneil/CalSciPy?branch=master)
[![Documentation Status](https://readthedocs.org/projects/calscipy/badge/?version=latest)](https://calscipy.readthedocs.io/en/latest/?badge=latest)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/CalSciPy?)
![PyPI - Status](https://img.shields.io/pypi/status/CalSciPy)
![GitHub](https://img.shields.io/github/license/darikoneil/CalSciPy)
![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/darikoneil/CalSciPy/calscipy_lint_test_action.yml)

CalSciPy is a utility toolbox for calcium imaging experiments. It contains a variety of useful features, from 
interactive visualization of data to computer-generated holography for "read/write" experiments, and 
everything in-between. Essentially, it's a collection of code written for my imaging experiments that might be 
considered useful to others. I was inspired to upload my own code to solve some of the issues outlined in the [motivation](https://calscipy.readthedocs.io/en/latest/introduction__motivation.html) section of the [docs](https://calscipy.readthedocs.io/en/latest/index.html#) or at least create a neat package my friends and I could use to easily analyze data across various environments and computers.


#### Active Development
As of version 0.9.8, the current implementation is unstable, partially untested, partially finished, and should be considered an open alpha/beta. ***Please be patient, refactoring my code for public use is a pet-project. I have to graduate at some point and I have a newborn.***

Until things are more stable, I'll explicitly note which subpackages are stable, have >90% test coverage, and are ready-to-use.

#### Stable & Tested Modules
* The main module is ready to use as of 0.7.5. New features will be demarcated by a label indicating the version added in the docs.
#### Experimental
* The organization module is untested but incorporated into the distributed pypi version
* The bruker module is untested but incorporated into the distributed pypi version
* The optogenetics module is untested but incorporated into the distributed pypi version
  
#### Unstable
* The interactive module is unstable but partially incorporated into the distributed pypi version
* The optics module is unstable and not incorporated into the distributed pypi version
  
#### Installation
Eventually I will break things up into sub-packages so you don't have to install everything together...         
`pip install CalSciPy`

#### Contributions
Save me from myself, contributions welcome :)

#### Documentation
Hosted at [ReadtheDocs](https://calscipy.readthedocs.io/en/latest/index.html#).
Available locally as [HTML](https://github.com/darikoneil/CalSciPy/tree/master/docs/build/html), [LATEX](https://github.com/darikoneil/CalSciPy/tree/master/docs/build/latex) and [PDF](https://github.com/darikoneil/CalSciPy/blob/master/docs/build/pdf/calscipy.pdf).
