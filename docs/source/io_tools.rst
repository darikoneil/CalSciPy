Loading and Saving Data
=======================
Using CalSciPy you can load, convert, and save your imaging data using very simple syntax.


Images (.tif)
*************

Loading
```````
CalScipy offers a single, simple function for loading images with the .tif file format and other closely associated
formats like .ome.tiff.

**Loading a 2D-image**
To load a two-dimensional image, simply pass the file as an argument to
:func:`load_images`<CalSciPy.io_tools.load_images>

.. code-block:: python

   images = load_images("single_image.tif")

**Loading a 3D-stack**
To load a three-dimensional image, simply pass the file as an argument to
:func:`load_images`<CalSciPy.io_tools.load_images>

.. code-block:: python

   images = load_images("imaging_stack.tif")

**Loading entire folders**
To load an entire folder full of images, simply pass the file as an argument to
:func:`load_images`<CalSciPy.io_tools.load_images>

.. code-block:: python

   images = load_images("imaging_folder")

Easy, eh?

Saving
``````
CalScipy also offers a single, simple function for saving images with the .tif file format.

**Saving a 2D-image**
To save a two-dimensional image, simply pass the file and a :class:`numpy array`<numpy.ndarray> as arguments to
:func:`save_images`<CalSciPy.io_tools.save_images>

.. code-block:: python

   images = save_images("single_image.tif", images)

**Saving a 3D-stack**
To save a three-dimensional image, simply pass the file and a :class:`numpy array`<numpy.ndarray> as arguments to
:func:`save_images`<CalSciPy.io_tools.save_images>

.. code-block:: python

   images = save_images("imaging_stack.tif", images)


Binary (Recommended)
********************
Binary

Video (.mp4)
************
Videos

Animations (.gif)
*****************
Animations
