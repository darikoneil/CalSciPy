
.. |ss| raw:: html

   <strike>

.. |se| raw:: html

   </strike>

Motivation
**********
I was inspired to upload my own code to solve some of these issues--or at least create neat package my friends and I
could use to easily analyze data across various environments and computers.

Turn "one-liners" into one-liners
`````````````````````````````````
I noticed I was constantly re-writing boilerplate, "one-liners", and copy/pasting code between
environments. I wrote this so |ss| I don't have to |se| you don't have to. It's intended to be a collection of useful,
well-documented functions often used in boilerplate code alongside software packages such as
`Caiman <https://github.com/flatironinstitute/CaImAn>`_, `SIMA <https://github.com/losonczylab/sima>`_,
and `Suite2P <https://github.com/MouseLand/suite2p>`_. Examples of such tasks include compiling imaging stacks,
normalizing signals to baselines, reorganizing data, and common data visualizations. No more wasting time writing 6
lines to simply preview your tiff stack, extract a particular channel, or bin some spikes. No more vague tracebacks or
incomplete documentation when re-using a hastily-made function from 2 months ago.

Data-centric interactive visualizations
```````````````````````````````````````
Most scientific software these days have GUIs. But if you want to visualize your data after performing some arbitrary
transformation? *Womp womp*. Maybe you use different tools to extract the same sorts of data, and just
want one darn gui to look at the both? *Womp womp* Well, I want generic, interactive visualization tools. I want to
assess the quality of spike inference from recently published spike-finder number #12, I want to sort through ROIs,
and I want to explore population activity, and I want to do so directly from the relevant data--independent of the
software used to collect or analyze it. I want to have my delicious cake and eat it too. So I made some interactive
visualization tools that require only the exact information that ought to be required to plot them.

Optics for neuroscientists
``````````````````````````
I needed tools for holographic optogenetics & "read/write" experimentation in python for my own experiments.
Since I was already doing it, why not make them abstracted enough for all neuroscientists to use...not for only the
optical or imaging experts. I want them to be tested, documented, and easy-to-use.

Pythonizing PrairieView
```````````````````````
I needed some more bespoke software for use with Bruker's PrairieView imaging software for tasks such as parsing
metadata, generating protocols programmatically (I'm lazy), and aligning collected data.

Simplified data organization & analysis
```````````````````````````````````````
Finally, I wanted a flexible & extensible system for neatly organizing data, timestamping analysis, & single-step
analysis.
