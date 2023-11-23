Image Processing
================

Resonant Deinterlacing
**********************
Resonant scanning microscopes achieve fast image acquisition by pairing a slow-galvometric mirror with a fast-resonant
scanning mirror: the fast-resonant scanning mirror rapidly scans a single-axis of the field-of-view
(i.e., a horizontal line), while the slow-galvometric mirror moves the line along a second-axis.
Further gains in acquisition speed are achieved by acquired images during both the forward and backward scans of the
resonant mirror. That is, lines are scanned in alternating directions rather than returning to the origin to scan each
line in a left-right, left-right, left-right strategy. While galvometric scanning mirrors can be tightly-controlled
using servos, resonant scanning microscopes achieve their rapid motion by vibrating at a fixed frequency in response
to applied voltage. Resonant scanning are extremely under-dampened harmonic oscillators; while their frequency is
tightly fixed, they are prone to large variations in amplitude given small deviations in their drive. Resonant
scanners also display a smooth cosinusoidal velocity through their range-of-motion--the scanner moves fastest in
the center and slower on the edges--that further complicates synchronizing to an external frequency. Therefore, the
entire microscope is typically aligned to the motion of the resonance scanner.

.. image:: scan_sync.png
    :width: 800

Most microscopy software organized the
incoming pixel-stream into resonant scanner cycles. Rather than immediately organizing pixels into individual lines,
pixels are collected through a complete oscillation of the resonant scanner. The data is then split down the center
and the latter half reversed to generate two lines. Because the position of the mirror is variable, the index of center
pixel is usually offset within the software or in real-time. Never-the-less, variations related to temperature,
voltage, and murphy's law--as well as poor signal-to-noise--often result in images with interlacing artifacts.
CalSciPy provides a convienent deinterlacing function to correct for these artifacts

.. centered:: Deinterlacing images

.. code-block:: python

   from CalSciPy.images import deinterlace
   import numpy as np

   images = deinterlace(images)

   # in-place
   images = deinterlace(images, in_place=True)

   # significant memory constraints
   batch_size = 5000 # number of frames to include in one batch
   images = deinterlace(images, batch_size=batch_size, in_place=True)

   # correcting noisy data with external reference
   y_pixels, x_pixels = images.shape[1:]
   reference = np.std(images, axis=0).reshape(1, y_pixels, x_pixels)  # get z-projected standard deviation
   images = deinterlace(images, reference=reference)


Multi-dimensional Filtering
***************************
CalSciPy supports fast de-noising of imaging stacks using multidimensional filters.

.. centered:: Filtering imaging stacks

.. code-block:: python

   from CalSciPy.images import gaussian_filter

   # standard deviation of gaussian kernel
   sigma = 1.0

   filtered_images = gaussian_filter(images, sigma=sigma)

In some situations you may be under memory-constraints. CalScipy supports both in-place and blockwise filtering in these
scenarios: simply utilize the in_place or block_size keywords.

.. centered:: Memory-constrained filtering

.. code-block:: python

   from CalSciPy.images import median_filter

   # size of median filter
   window = (3, 3, 3)

   # 7000 frame blocks
   filtered_images = median_filter(images, window=window, block_size=7000)

   # 7000 frame blocks with 3500 frame overlap
   filtered_images = median_filter(images, window=window, block_size=7000, block_buffer=3500)

   # in-place calculation
   filtered_images = median_filter(images, window=window, in_place=True)


Available Multi-dimensional Filters
***********************************
* :func:`Gaussian Filter <CalSciPy.images.gaussian_filter>`
* :func:`Median Filter <CalSciPy.images.median_filter>`


.. note::

   Using gpu-parallelization is recommended to quickly process imaging stacks. Being said, using gpu parallelization
   requires that the dataset fit within your GPU's VRAM. In most cases, this requires breaking the dataset down into
   smaller blocks. This can be done automatically by using the block_size keyword.
