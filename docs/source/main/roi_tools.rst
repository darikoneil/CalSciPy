Regions of Interest
===================
The :mod:`roi_tools <CalSciPy.roi_tools>` module provided a standardized :class:`ROI <CalSciPy.roi_tools.ROI>` class
to represent regions of interest in your imaging data. By having this standardized class, we can import ROIs from any
arbitrary data structure. That is, we can load ROIs from any analysis software we want. A dictionary of ROIs can be
easily formed using the default or custom :class:`ROI Handlers <CalSciPy.roi_tools.ROIHandler>`. Software-specific
:class:`ROI Handlers <CalSciPy.roi_tools.ROIHandler>` are available for many major calcium imaging software packages
and a :class:`metaclass <CalSciPy.roi_tools.ROIHandler>` is provided for customizing your own.

ROI instances
`````````````
Each :class:`ROI <CalSciPy.roi_tools.ROI>` instance contains the basic characteristics & properties of an ROI.
Constructing an ROI requires only knowledge of its pixel coordinates. From these coordinates, a variety of attributes
are calculated, including its centroid, radius, and approximate outline. Importantly, these attributes are only
calculated at instantiation and then permanently cached for performance benefits. Therefore, changing the pixels of an
existing :class:`ROI <CalSciPy.roi_tools.ROI>` instance is not permitted. A properties attribute is provided to allow
users to incorporate arbitrary information for a particular ROI (e.g., SNR). All keyword arguments will be passed to
the properties attribute.

.. centered:: **Generating an ROI instance**

.. code-block:: python

   import numpy as np
   from CalSciPy.roi_tools import ROI

   x_pixels = [31, 31, 31, 32, 32, 32, 33, 33, 33]
   y_pixels = [31, 32, 33, 31, 32, 33, 31, 32, 33]
   reference_shape = (64, 64)
   properties = {"cool_roi": True}

   roi = ROI(x_pixels, y_pixels, reference_shape, properties, special_roi=True)

   >>>print(f"{roi.radius=}")
   roi.radius=1

   >>>print(f"{roi.centroid=}")
   roi.centroid=(32, 32)

   >>>print(f"{roi.properties}")
   roi.properties={'special_roi': True, 'cool_roi': True}

Importing ROIs from File
````````````````````````
The :class:`ROI Handler <CalSciPy.roi_tools.ROIHandler>` metaclass describes a standardized approach to loading
ROIs and can be adapted to handle any arbitrary data structure your ROIs may be stored in. Calling the
:func:`load <CalSciPy.roi_tools.ROIHandler.load>` class method for any ROIHandler will generate a tuple
containing a dictionary of :class:`ROIs <CalSciPy.roi_tools.ROI>` and a reference image in the form of
a :class:`numpy array <numpy.ndarray>` with shape height x width. Each key of the dictionary is an integer indexing
the ROI and its value-pair is the associated :class:`ROI <CalSciPy.roi_tools.ROI>`.
:class:`ROI Handlers <CalSciPy.roi_tools.ROIHandler>` for common analysis software are provided.

Suite2P
```````
The :class:`Suite2P Handler <CalSciPy.roi_tools.suite2p_handler.Suite2PHandler>` is provided for loading
`suite2p <https://www.suite2p.org>`_ data. It requires the path to a folder containing suite2p data
as its only argument. The folder ought to contain at least the *stat.npy* and *ops.npy* files, although the
*iscell.npy* file is also recommended.

.. centered:: **Using the Suite2P Handler**

.. code-block:: python

   from CalSciPy.roi_tools import Suite2PHandler

   rois, reference_image = Suite2PHandler("file_location")
