Visuals
============

CalSciPy Style
``````````````
Access to the styles utilized by CalSciPy is provided through :class:`COLORS <CalSciPy.color_scheme.COLORS>`\,
:class:`TERM_SCHEME <CalSciPy.color_scheme.TERM_SCHEME>`\, ColorSpectrum, and through matplotlib's style contexts.

Color Scheme
############
Access to :class:`COLORS <CalSciPy.color_scheme.COLORS>` permits retrieval of the associated RGB tuple for a desired
color through its name, index, or directly through its attribute implementation. More information on which colors are
available can be found :class:`here <CalSciPy.color_scheme._ColorScheme>`\.

.. centered:: **Using CalSciPy's color scheme**

.. code-block:: python

    from CalSciPy.color_scheme import COLORS

    red = COLORS.RED

    red = COLORS("red")

    red = COLORS(0)

    red, green, blue = [COLORS(i) for i in range(3)]

Matplotlib Style
################
Users can utilized CalSciPy's matplotlib style through matplotlib style context manager.

.. centered:: **Using CalSciPy's matplotlib style**

.. code-block:: python

    import matplotlib
    from matplotlib import pyplot as plt

    with plt.style.context("CalSciPy.main"):
        fig = plt.figure()
