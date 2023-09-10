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
considered useful to others. Its inspiration is to solve the following issues:

Motivation: I was inspired to upload my own code to solve some of these issues--or at least create a neat package my friends and I 
could use to easily analyze data across various environments and computers.

1. I noticed I was constantly re-writing boilerplate, "one-liners", and copy/pasting code between
environments. I wrote this so ~I don't have to~ you don't have to. It's intended to be a collection of useful, 
well-documented functions often used in boilerplate code alongside software packages such as 
[Caiman](https://github.com/flatironinstitute/CaImAn), [SIMA](https://github.com/losonczylab/sima), 
and [Suite2P](https://github.com/MouseLand/suite2p). Examples of such tasks include compiling imaging stacks, 
normalizing signals to baselines, reorganizing data, and common data visualizations. No more wasting time writing 6
lines to simply preview your tiff stack, extract a particular channel, or bin some spikes. No more vague tracebacks or
incomplete documentation when re-using a hastily-made function from 2 months ago.

2. Most scientific software these days have GUIs. But if you want to visualize your data after performing some arbitrary
transformation? *Womp womp*. Maybe you use different tools to extract the same sorts of data, and just
want one darn gui to look at them both? *Womp womp* Well, I want generic, interactive visualization tools. I want to
assess the quality of spike inference from recently published spike-finder number #12, I want to sort through ROIs,
and I want to explore population activity. I want to do so directly from the relevant data--independent of the
software used to collect or analyze it. I want to have my delicious cake and eat it too. So I made some interactive
visualization tools that require only the exact information that ought to be required to plot them.

3. I needed tools for holographic optogenetics, "read/write" experimentation, and general optics in python for my own experiments.
Since I was already doing it, why not make them abstracted enough for all neuroscientists to use...not for only the
optical or imaging experts. I want them to be tested, documented, and easy-to-use.

5. I needed some more bespoke software for use with Bruker's PrairieView imaging software for tasks such as parsing 
metadata, generating protocols programmatically, and aligning collected data.

6. Finally, I wanted a flexible & extensible system for neatly organizing & timestamping data.


#### Active Development
The current implementation is unstable, partially untested, partially finished, and should be considered an open 
alpha/beta. Please be patient, refactoring my code for public use is a pet-project. I have to graduate at some point.


#### Stable Modules
Until things are more stable, I'll explicitly note which modules' are stable, have >90% test coverage, and are
ready-to-use.
* io_tools as of version 0.4.0


#### Installation
Eventually I will break things up into sub-packages so you don't have to install everything together...         
`pip install CalSciPy`

#### Contributions
Save me from myself

#### Documentation
Hosted at [ReadtheDocs](https://calscipy.readthedocs.io/en/latest/index.html#).
Available locally as [HTML](https://github.com/darikoneil/CalSciPy/tree/master/docs/build/html), [LATEX](https://github.com/darikoneil/CalSciPy/tree/master/docs/build/latex) and [PDF](https://github.com/darikoneil/CalSciPy/blob/master/docs/build/pdf/calscipy.pdf).
