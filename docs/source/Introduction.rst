
.. |ss| raw:: html

   <strike>

.. |se| raw:: html

   </strike>

Introduction
============
**CalSciPy** contains a variety of useful methods for handling, processing, and visualizing calcium imaging data.
It's intended to be a collection of useful, well-documented functions often used in boilerplate code alongside software
packages such as `Caiman <https://github.com/flatironinstitute/CaImAn>`_, `SIMA <https://github.com/losonczylab/sima>`_,
and `Suite2P <https://github.com/MouseLand/suite2p>`_.

Motivation
**********
I noticed I was often re-writing or copy/pasting a lot of code between environments when working with calcium imaging
data. I started this package |ss| so I don't have to |se| so you don't have to. No more wasting time writing 6 lines to simply
preview your tiff stack, extract a particular channel, or bin some spikes. No more vague exceptions or incomplete
documentation when re-using a hastily-made function from 2 months ago. Alongside these time-savers, I've also included
some more non-trivial methods that are particularly useful.

Limitations
***********
The current distribution for the package is incomplete and partially tested. There may be breaking changes between versions.