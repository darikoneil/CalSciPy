Image Processing
================

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





