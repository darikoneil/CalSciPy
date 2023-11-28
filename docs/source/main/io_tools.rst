Loading and Saving Data
=======================
You can load, convert, and save your imaging data using very simple syntax through functions in the
:mod:`io_tools <CalSciPy.io_tools>` module.

Images (.tif)
*************

Loading .tif's
``````````````
CalScipy offers a single, simple function for loading images with the *.tif* file format and other closely associated
formats like *.ome.tiff*. The :func:`load_images <CalSciPy.io_tools.load_images>` function loads *.tif* images into
:class:`numpy arrays <numpy.ndarray>` with shape *frames* x *y-pixels* x *x-pixels*. It requires the path to the images
as an argument.

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
CalScipy also offers a single, simple function for saving images with the *.tif* file format. To save images,
simply pass the desired location and a :class:`numpy array <numpy.ndarray>` as arguments to
:func:`save_images <CalSciPy.io_tools.save_images>`\. If the image is size is larger the *size_cap* limit, the stack
will be automatically split into chunks of size *size_cap*. By default, the size_cap is set to limit *.tif* stacks to
less than 4GB each for compatibility with the majority of *.tif* readers.

.. centered:: **Saving images to file**

.. code-block:: python

   save_images("single_image.tif", images)

.. centered:: **Saving images to a folder**

.. code-block:: python

   save_images("desired_folder", images)

.. centered:: **Saving images to a folder with specified name**

.. code-block:: python

   save_images("desired_folder", images, name="example_images")

Binary (Recommended)
********************
CalScipy offers functions for loading and saving images as binary. This language-agnostic format is ideal for optimal
read/write speeds, larger-than-memory data, and is highly-robust to corruption. However, it does have downsides. First,
the images and their metadata are split into two separate files: ".bin" and ".json" respectively. If you happen to lose
the metadata file, fear not! As long as you have the datatype and 2 of the 3 dimensions you can still load the data.
A second disadvantage is a lack of compression. Using binary is excellent in cases where storage space is "cheaper" than
I/O time: for example, when data is still being regularly accessed and not simply sitting in "cold storage".

Loading binary
``````````````
Binary data in CalSciPy can be loaded using the :func:`load_binary <CalSciPy.io_tools.load_binary>` function.
The path to the binary data is required as an argument and the data is returned as a
:class:`numpy array <numpy.ndarray>` with shape *frames* x *y-pixels* x *x-pixels*.

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

Saving binary
`````````````
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

Loading video
`````````````
Loading .mp4's uses the :func:`load_video <CalSciPy.io_tools.load_video>` function, returning the video as
a :class:`numpy array <numpy.ndarray>` with shape *frames* x *y-pixels* x *x-pixels* x *color channels*

.. centered:: **Loading video from file**

.. code-block:: python

    images = load_video("video_file.mp4")

.. centered:: **Loading video from folder**

.. code-block:: python

    images = load_video("desired_folder")

Saving video
````````````
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
