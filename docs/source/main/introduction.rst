
.. |ss| raw:: html

   <strike>

.. |se| raw:: html

   </strike>

Introduction
============

I created the main module of CalSciPy because I noticed I was constantly re-writing boilerplate, "one-liners", and
copy/pasting code between environments. I wrote this so |ss| I don't have to |se| you don't have to. It's intended to be
a collection of useful, well-documented functions often used in boilerplate code alongside software packages such as
`Caiman <https://github.com/flatironinstitute/CaImAn>`_, `SIMA <https://github.com/losonczylab/sima>`_,
and `Suite2P <https://github.com/MouseLand/suite2p>`_. Examples of such tasks include compiling imaging stacks,
normalizing signals to baselines, reorganizing data, and common data visualizations. No more wasting time writing 6
lines to simply preview your tiff stack, extract a particular channel, or bin some spikes. No more vague tracebacks or
incomplete documentation when re-using a hastily-made function from 2 months ago.

Main contains modules for I/O operations on imaging data, processing imaging stacks, processing fluorescent traces,
processing inferred spikes or events, a standardized format for defining region-of-interests, and an object-oriented
approach to optogenetic experiments. The following pages highlight some of the main modules capabilities and standard
use cases. Comprehensive documentation can be found in the API reference.
