
.. |ss| raw:: html

   <strike>

.. |se| raw:: html

   </strike>

Motivation
**********

1. I noticed I was constantly re-writing boilerplate, "one-liners", and copy/pasting code between
environments. I wrote this so |ss| I don't have to |se| you don't have to. It's intended to be a collection of useful,
well-documented functions often used in boilerplate code alongside software packages such as
`Caiman<https://github.com/flatironinstitute/CaImAn>`_, `SIMA<https://github.com/losonczylab/sima>`_,
and `Suite2P<https://github.com/MouseLand/suite2p>`_. Examples of such tasks include compiling imaging stacks,
normalizing signals to baselines, reorganizing data, and common data visualizations. No more wasting time writing 6
lines to simply preview your tiff stack, extract a particular channel, or bin some spikes. No more vague tracebacks or
incomplete documentation when re-using a hastily-made function from 2 months ago.

2. I want generic, interactive visualization tools. I want to assess the quality of spike inference, I want to sort
through ROIs, and I want to explore population activity, and I want to do so directly from the relevant
data--independent of the software used to analyze it. I want to have my delicious cake and eat it too.

3. I want tools for holographic optogenetics & "read/write" experimentation in python. I want them written for
neuroscientists to use...not for optical or imaging experts. I want them to be tested, documented, and easy-to-use.
It's 2023.

4. I needed some more bespoke software for use with Bruker's PrairieView imaging software for tasks such as parsing
metadata, generating protocols programmatically (I'm lazy), and aligning collected data.

5. Finally, I wanted a flexible & extensible system for neatly organizing & timestamping data.

I was inspired to upload my own code to solve these issues--or at least create neat package my friends and I could use
to easily analyze data across various environments and computers.
