Trace Processing
================

Calculating Δf/f0
*****************
CalSciPy supports a variety of methods for calculating the fold fluorescence over baseline through a single function.

.. code-block:: python

   from CalScipy.traces import calculate_dfof

   dFoF = calculate_dfof(traces, method="mean")


Supported baseline calculation methods
``````````````````````````````````````
* "low-pass": x-th percentile of the :func:` low-pass <CalSciPy.traces.baseline.low_pass_baseline>` filtered trace
* "mean": :func:`mean <CalSciPy.traces.baseline.mean_baseline>` of the trace
* "median": :func:`median <CalSciPy.traces.baseline.median_baseline>` of the trace
* "moving_mean": :func:`moving mean <CalSciPy.traces.baseline.median_baseline>` of the trace using a specified window length
* "percentile": :func:`x-th percentile <CalSciPy.traces.baseline.percentile_baseline>` of the trace
* "sliding_mean": :func:`sliding mean <CalSciPy.traces.baseline.sliding_mean_baseline>` calculated using a sliding window of specified length
* "sliding_median: :func:`sliding median <CalSciPy.traces.baseline.sliding_median_baseline>` calculated using a sliding window of specified length
* "sliding_percentile: :func:`sliding percentile <CalSciPy.traces.baseline.sliding_percentile_baseline>` calculated using a sliding window of specified length

Baseline Corrections
********************
CalSciPy currently provides a function for polynomial detrending to correct for a drifting baseline due to
time-dependent degradation of signal-to-noise

.. code-block:: python

   from CalScipy.traces import detrend_polynomial

   detrended_traces = detrend_polynomial(traces, frame_rate=frame_rate)

Assessing Trace Quality
***********************
CalSciPy supports assessment of trace quality using a standardized noise metric first defined in the publication
associated with the spike-inference software package `CASCADE <https://www.nature.com/articles/s41593-021-00895-5>`_\.

.. code-block:: python

   from CalScipy.traces import calculate_standardized_noise

   # acquisition frequency
   frame_rate = 30
   standardized_noise = calculate_standardized_noise(dFoF, frame_rate=frame_rate)

