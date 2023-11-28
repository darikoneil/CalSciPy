Getting Started
===============

Installation
************
Eventually I will break the distributions up so you don't have to necessarily install some of the more tangential
modules. For right now, everything is in one package.

Enter ``pip install CalSciPy`` in your terminal.

If you want to use gpu-parallelization, you will need to install `CuPy <https://github.com/cupy/cupy>`_
by following these `instructions <https://docs.cupy.dev/en/stable/install.html>`_.

Limitations
***********
The current implementation is unstable, partially untested, partially finished, and should be considered an open
alpha/beta. Please be patient, this is a pet-project, I have a newborn, and I need to finish my thesis.
Until things are more finalized, I'll explicitly note which modules are stable, have >90% test coverage, and are
ready-to-use. Anything not listed as table should be considered experimental and use at your own risk. If you really
need something labeled experimental I can probably provide offline code that will probably prove more useful short-term.
While the majority of features are already written, they need to be refined for maintainability, readability, and to
remove any bugs.

Stable Modules
##############
The following modules are stable as of the indicated version. New features added will be demarcated with a
version added tag in the documentation and there will be no further breaking changes.

* :mod:`color_scheme <CalSciPy.color_scheme>` 0.7.5
* :mod:`events <CalSciPy.events>` 0.7.5
* :mod:`images <CalSciPy.images>` 0.7.5
* :mod:`io_tools <CalSciPy.io_tools>` 0.7.5
* :mod:`traces <CalSciPy.traces>` 0.7.5

Unstable Modules
################
These modules are considered unstable. They may be experimental, undocumented, untested, or unfinished.

* :mod:`bruker <CalSciPy.bruker>`
* :mod:`conversion <CalSciPy.conversion>`
* :mod:`interactive <CalSciPy.interactive>`
* :mod:`optics <CalSciPy.optics>`
* :mod:`optogenetics <CalSciPy.optogenetics>`
* :mod:`organization <CalSciPy.organization>`
* :mod:`roi_tools <CalSciPy.roi_tools>`
