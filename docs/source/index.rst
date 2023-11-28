.. image:: dancing_neuron_band.png
    :width: 800

CalSciPy
====================================
**CalSciPy** is a utility toolbox for calcium imaging experiments. It contains a variety of useful features, from
interactive visualization of data to computer-generated holography for "read/write" experiments, and
everything in-between. Essentially, it's a collection of code written for my imaging experiments that might be
considered useful to others.

Motivation
**********
While working on my Ph.D. I noticed I was constantly re-writing boilerplate, "one-liners" and copying/pasting
more-extensive code between environments. Ever try to use a minimally-documented function written during an exploratory
analysis on a different operating system two years later? How much time have you spent debugging hardcoded variables
after adapting functions from one study to the next? During some lunchtime reflection, I realized that a lot of my
colleagues were simultaneously writing (and re-writing) very similar code to mine. How much time have scientists spent
reinventing the wheel? Why did each of us write our own GUI for interacting looking at traces? And why do they all rely
unstandardized, idiosyncratic organizations of datasets. I decided to spend a little extra time thoroughly
documenting and testing to put together this package so at least my friends and I could easily analyze data across
various environments, interpreters and operating systems. While having all your code so well documented and tested
takes more time--I found it does wonders for your stress levels (documenting is a relaxing monotonous activity) and
people like you more. Plus, all the abstraction and documentation makes your analysis easy to read and follow.
As my programming skills and scientific expertise have grown, I've started to roadmap the inclusion of some more complex
functionality--*refactoring* my existing code of interest, *writing* code for my projects with future distribution in
mind, and *implementing* new features the science community could benefit from--because making scientific software
open-source and proliferating expertise is simply the right thing to do.

Highlights
**********

* Useful, well-documented functions often used in boilerplate code alongside registration and roi extraction software packages such as `Caiman <https://github.com/flatironinstitute/CaImAn>`_, `SIMA <https://github.com/losonczylab/sima>`_, and `Suite2P <https://github.com/MouseLand/suite2p>`_.
* Flexible & extensible system for neatly organizing data, timestamping analysis, & single-step analyses.
* Quality-of-life utilities for Bruker's PrairieView software
* Interactive data visualization
* Tools for optics & optogenetics


.. toctree::
   :maxdepth: 1
   :caption: CONTENTS
   :hidden:

    Getting Started <getting_started>
    Loading/Saving Images <main/io_tools>
    Image Pre-Processing <main/images>
    Trace Processing <main/traces>
    Events and Spiking <main/events>
    Arranging and Aligning Data <main/conversion>
    Regions of Interest <main/roi_tools>
    Optogenetics <main/optogenetics>
    Style <main/style>
    Dependencies & Contributions <contributions>
    API Reference <api_reference>
