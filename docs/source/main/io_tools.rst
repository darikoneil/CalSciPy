Loading and Saving Data
=======================
You can load, convert, and save your imaging data using very simple syntax through functions in the
:mod:`io_tools <CalSciPy.io_tools>` module.

Images (.tif)
*************

Loading
```````
CalScipy offers a single, simple function for loading images with the .tif file format and other closely associated
formats like .ome.tiff. The :func:`load_images <CalSciPy.io_tools.load_images>` function loads .tif images into
:class:`numpy arrays <numpy.ndarray>` with shape frames x height x width.
It requires the path to the images as an argument.

.. centered:: **Loading a 2D-image**

.. code-block:: python

   images = load_images("single_image.tif")

.. centered:: **Loading a 3D-stack**

.. code-block:: python

   images = load_images("imaging_stack.tif")

.. centered:: **Loading entire folders**

.. code-block:: python

   images = load_images("imaging_folder")

Easy, huh?

Saving
``````
CalScipy also offers a single, simple function for saving images with the .tif file format. To save images,
simply pass the desired location and a :class:`numpy array <numpy.ndarray>` as arguments to
:func:`save_images <CalSciPy.io_tools.save_images>`

.. centered:: **Saving images to file**

.. code-block:: python

   save_images("single_image.tif", images)

.. centered:: **Saving images to a folder**

.. code-block:: python

   save_images("desired_folder", images)

.. centered:: **Saving images as multiple stacks**

.. code-block:: python

    save_images("desired_folder", images, size_cap=0.01)

.. centered:: **Saving images to a folder with specified name**

.. code-block:: python

   save_images("desired_folder", images, name="example_images")

Binary (Recommended)
********************
CalScipy offers functions for loading and saving images as binary. This language-agnostic format is ideal for optimal
read/write speeds, larger-than-memory data, and is highly-robust to corruption. However, the downside is that the
images and their metadata are split into two separate files: ".bin" and ".json" respectively. If you happen to lose the
metadata file, fear not! As long as you have the datatype and 2 of the 3 dimensions you can still load the data.

Loading
```````
Binary data in CalSciPy can be loaded using the :func:`load_binary <CalSciPy.io_tools.load_binary>` function.
The path to the binary data is required as an argument and the data is returned as a :class:`numpy array <numpy.ndarray>`
with shape frames x height x width.

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

Saving
``````
Saving data to binary in CalSciPy uses the :func:`save_binary <CalSciPy.io_tools.save_binary>` function with the
file path and a :class:`numpy array <numpy.ndarray>` as arguments

.. centered:: **Saving binary to file**

.. code-block:: python

    save_binary("binary_file.bin", images)

.. centered:: **Saving binary to folder**

.. code-block:: python

    save_binary("desired_folder", images)

.. centered:: **Saving binary to folder with specified name**

.. code-block:: python

    save_binary("desired_folder", images, name="example_binary")

Video (.mp4)
************
CalSciPy also provides simple functions to load and save .mp4 files.

Loading
```````
Loading .mp4's uses the :func:`load_video <CalSciPy.io_tools.load_video>` function, returning the video as
a :class:`numpy array <numpy.ndarray>` with shape frames x height x width x color channel

.. centered:: **Loading video from file**

.. code-block:: python

    images = load_video("video_file.mp4")

.. centered:: **Loading video from folder**

.. code-block:: python

    images = load_video("desired_folder")

Saving
``````
Saving .mp4's uses the :func:`save_video <CalSciPy.io_tools.save_video>` function with a file path and
a :class:`numpy array <numpy.ndarray>` as arguments.

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
