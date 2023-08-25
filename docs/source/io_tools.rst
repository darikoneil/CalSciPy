Loading and Saving Data
=======================
Using CalSciPy you can load, convert, and save your imaging data using very simple syntax.


Images (.tif)
*************

Loading
```````
CalScipy offers a single, simple function for loading images with the .tif file format and other closely associated
formats like .ome.tiff. The :func:`load_images <CalSciPy.io_tools.load_images>` function loads .tif images into
numpy arrays with shape frames x height x width. It requires the path to the images as an argument.

**Loading a 2D-image**

.. code-block:: python

   images = load_images("single_image.tif")

**Loading a 3D-stack**

.. code-block:: python

   images = load_images("imaging_stack.tif")

**Loading entire folders**

.. code-block:: python

   images = load_images("imaging_folder")

Easy, eh?

Saving
``````
CalScipy also offers a single, simple function for saving images with the .tif file format. To save images, simply pass the file and a :class:`numpy array <numpy.ndarray>` as arguments to
:func:`save_images <CalSciPy.io_tools.save_images>`

**Saving images**

.. code-block:: python

   images = save_images("single_image.tif", images)



Binary (Recommended)
********************
Binary

Video (.mp4)
************
Videos

Animations (.gif)
*****************
Animations
