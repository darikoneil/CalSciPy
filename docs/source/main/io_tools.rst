Loading and Saving Data
=======================
Simple importing and exporting of imaging data is a significant source of boilerplate code in most processing pipelines.
The :mod:`io_tools <CalSciPy.io_tools>` module provides a few functions for loading, converting, and saving imaging
data with very simple syntax and good performance. Currently, *\*.tif*, *\*.gif*, *\*.mp4*, and *\*.bin* (binary) file
types are supported. Every function either returns or expects the images in the form of a
:class:`numpy arrays <numpy.ndarray>` with shape *frames*, *y-pixels*, *x-pixels*. They also all use a similar syntax:
**load_{file_type}**\(path) for loading and **save_{file_type}**\(path, images) for saving.

.. note::
    Color videos and gifs additionally accept :class:`numpy arrays <numpy.ndarray>` with
    shape *frames*, *y-pixels*, *x-pixels*, *color*.

Loading .tif's
``````````````
CalScipy offers a single, simple function for loading images with the *\*.tif* file format and other closely associated
formats like *\*.ome.tiff*: :func:`load_images <CalSciPy.io_tools.load_images>`\.

.. centered:: **Loading a 2D-image**

.. code-block:: python

   images = load_images("single_image.tif")

.. centered:: **Loading a 3D-stack**

.. code-block:: python

   images = load_images("imaging_stack.tif")

.. centered:: **Loading entire folders of imaging stacks**

.. code-block:: python

   images = load_images("imaging_folder")

Easy, huh?

Saving .tif's
`````````````
CalScipy also offers a single, simple function for saving images with the *\*.tif* file format. If the image size is
larger the *size_cap* limit, the stack will be automatically split into chunks of size *size_cap*.
By default, the size_cap is set to limit *.tif* stacks to less than 4GB each for compatibility with the majority of
*\*.tif* readers.

.. centered:: **Saving images to file**

.. code-block:: python

   save_images("single_image.tif", images)

.. centered:: **Saving images to a folder**

.. code-block:: python

   save_images("desired_folder", images)

.. centered:: **Saving images to a folder with specified name**

.. code-block:: python

   save_images("desired_folder", images, name="example_images")

Loading .bin's
``````````````
Binary data in CalSciPy can be loaded using the :func:`load_binary <CalSciPy.io_tools.load_binary>` function with a
similar syntax. However, additional arguments are available to load the images without reading the entire file into
memory (i.e., memory-mapping).

.. centered:: **Loading binary data directly from file**

.. code-block:: python

    images = load_binary("binary.bin")

.. centered:: **Loading binary data directly from a folder**

.. code-block:: python

    images = load_binary("desired_folder")

.. centered:: **Loading memory mapped binary data**

.. code-block:: python

    images = load_binary("desired_folder", mapped=True, mode="r")

.. centered:: **Loading binary data with missing metadata**

.. code-block:: python

    missing_metadata = {"frames": 100, "y": 100, "dtype": int}
    images = load_binary("desired_folder", missing_metadata=missing_metadata)


Saving .bin's
`````````````
Binary data can be saved to file using the :func:`save_binary <CalSciPy.io_tools.save_binary` function.

.. centered:: **Saving binary to file**

.. code-block:: python

    save_binary("binary_file.bin", images)

.. centered:: **Saving binary to folder**

.. code-block:: python

    save_binary("desired_folder", images)

.. centered:: **Saving binary to folder with specified name**

.. code-block:: python

    save_binary("desired_folder", images, name="example_binary")

.. tip::

    This language-agnostic format is ideal for optimal read/write speeds, larger-than-memory data, and is highly-robust
    to corruption. However, it does have downsides. First, the images and their metadata are split into two separate
    files: ".bin" and ".json" respectively. If you happen to lose the metadata file, fear not! As long as you have the
    datatype and 2 of the 3 dimensions you can still load the data. A second disadvantage is a lack of compression.
    Using binary is excellent in cases where storage space is "cheaper" than I/O time: for example, when data is still
    being regularly accessed and not simply sitting in "cold storage".

Loading .mp4's
``````````````
Loading *\*.mp4*\'s uses the :func:`load_video <CalSciPy.io_tools.load_video>` function, returning the video as
a :class:`numpy array <numpy.ndarray>` with shape *frames*, *y-pixels*, *x-pixels*, *colors*.

.. centered:: **Loading video from file**

.. code-block:: python

    images = load_video("video_file.mp4")

.. centered:: **Loading video from folder**

.. code-block:: python

    images = load_video("desired_folder")

Saving .mp4's
`````````````
Saving *\*.mp4*\'s uses the :func:`save_video <CalSciPy.io_tools.save_video>` function. The frame rate of the video can be
set with the frame_rate argument.

.. centered:: **Saving video to file**

.. code-block:: python

    save_video("video_file.mp4", images)

.. centered:: **Saving video to folder**

.. code-block:: python

    save_video("desired_folder", images)

.. centered:: **Saving video to folder with specified name**

.. code-block:: python

    save_video("desired_folder", images, name="example_binary")

.. centered:: **Saving video to folder with specified framerate**

.. code-block:: python

    save_video("video_file.mp4", images, frame_rate=90.0)

Loading .gif's
``````````````
Loading *\*.gif*\'s uses the :func:`load_gif <CalSciPyt.io_tools.load_gif>` function.

.. centered:: **Loading a \*.gif**

.. code-block:: python

    gif = load_gif("gif_file.gif")

Saving .gif's
`````````````
Saving your images as a *\*.gif* is as easy as using the :func:`save_gif <CalSciPy.io_tools.save_gif>` function.

.. centered:: **Saving a \*.gif**

.. code-block:: python

    save_gif("gif_file.gif", images, frame_rate=5.0)

.. tip::

    Inserting videos into a presentation as a *\*.gif* is a clever way to avoid technical difficulties (shudder).
