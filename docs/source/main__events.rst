Events and Spiking
==================

.. note::

   Inferring spikes from calcium activity is a complex problem. Whether your "spikes" represent a single discrete
   spike--or a multi-spike event--depends on your choice of inference algorithm, the indicator, sampling rate,
   imaging quality, and innumerable other factors. It follows that the term *event* and *spike* are used
   interchangeably in CalSciPy.

Generating Rasters
``````````````````
To generate a raster, utilize the :func:`generate_raster <CalSciPy.events.generate_raster>` function.

Firing Rates
````````````
CalSciPy provides several small functions that abstract firing rate calculations and improve the readability of your
scripts.

* :func:`calc_firing_rates <CalSciPy.events.calc_firing_rates>`\: Calculate instantaneous firing rates from
  spike (event) probabilities.
* :func:`calc_mean_firing_rates <CalSciPy.events.calc_mean_firing_rates>`\: Calculate mean firing rates from \
  instantaneous firing rates.
* :func:`normalize_firing_rates <CalSciPy.events.normalize_firing_rates>`\: Normalize firing rates by scaling to a max
  of 1.0.
