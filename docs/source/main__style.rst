Style
============

CalSciPy Style
``````````````
Access to the styles utilized by CalSciPy is provided through :class:`COLORS <CalSciPy.color_scheme.COLORS>`\,
:class:`TERM_SCHEME <CalSciPy.color_scheme.TERM_SCHEME>`\, ColorSpectrum, and through
:class:`matplotlib's <matplotlib>` style contexts.

Color Scheme
############
Access to :class:`COLORS <CalSciPy.color_scheme.COLORS>` permits retrieval of the associated RGB tuple for a desired
color through its name, index, or directly through its attribute implementation. More information on which colors are
available can be found :class:`here <CalSciPy.color_scheme._ColorScheme>`\.

.. centered:: **Using CalSciPy's Color Scheme**

.. code-block:: python

    from CalSciPy.color_scheme import COLORS

    red = COLORS.RED

    red = COLORS("red")

    red = COLORS(0)

    red, green, blue = [COLORS(i) for i in range(3)]

Matplotlib Style
################
Users can utilize CalSciPy's :class:`matplotlib` style through
:class:`matplotlib's style context manager <matplotlib.pyplot.style.context>`\.

.. centered:: **Using CalSciPy's Matplotlib Style**

.. code-block:: python

    import matplotlib
    from matplotlib import pyplot as plt

    with plt.style.context("CalSciPy.main"):
        fig = plt.figure()

Terminal Style
##############
Users can utilize CalSciPy's terminal printing style by using :class:`TERM_SCHEME <CalSciPy.color_scheme.TERM_SCHEME>`\.
Calling :class:`TERM_SCHEME <CalSciPy.color_scheme.TERM_SCHEME>` with a specific property or attribute string alongside
a string message will return your original message with the requested formatting. More information on available
formatting can be found :class:`here <CalSciPy.color_scheme._TerminalScheme>`\.

.. centered:: **Using CalSciPy's Terminal Style

.. code-block:: python

   from CalSciPy.color_scheme import TERM_SCHEME

   message = "Hello World!"

   formatted_message = TERM_SCHEME(message, "header")
