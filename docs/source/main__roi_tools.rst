Regions of Interest
===================
The :mod:`roi_tools <CalSciPy.roi_tools>` module provided a standardized :class:`ROI <CalSciPy.roi_tools.ROI>` class
to represent regions of interest in your imaging data. By having this standardized class, we can import ROIs from any
arbitrary data structure. That is, we can load ROIs from any analysis software we want. A dictionary of ROIs can be
easily formed using the default or custom :class:`ROI Handlers <CalSciPy.roi_tools.ROIHandler>`. Software-specific
:class:`ROI Handlers <CalSciPy.roi_tools.ROIHandler>` are available for many major calcium imaging software packages
and a :class:`metaclass <CalSciPy.roi_tools.ROIBase>` is provided for customizing your own.

ROI instances
`````````````
Each :class:`ROI <CalSciPy.roi_tools.ROI>` instance contains the basic characteristics & properties of an ROI.
Constructing an ROI requires only knowledge of its pixel coordinates. From these coordinates, a variety of attributes
are calculated, including its centroid, radius, and approximate outline. Importantly, these attributes are only
calculated at instantiation and then permanently cached for performance benefits. Therefore, changing the pixels of an
existing :class:`ROI <CalSciPy.roi_tools.ROI>` instance is not permitted. A properties attribute is provided to allow
users to incorporate arbitrary information for a particular ROI (e.g., SNR).

Importing ROIs from file
````````````````````````
