.. _tutorial:

Tutorial
==========

In this tutorial, I will introduce the basic usage of ``pyfk``. The package ``pyfk`` can be divided into three parts, the configuration part, the Green's function calculation part, and the synthetic waveform generation part.

Configurations
-----------------

At the first step, we should generate three instances from the configuration classes including ``SourceModel``, ``SeisModel``, and ``Config``. For details of these classes, you can refer to :ref:`:mod:`pyfk.config.config``. Note the 
three classes have been imported into the upper-most level of the package, so you can directly import them:

.. jupyter-execute::

  from pyfk import SourceModel, SeisModel, Config

The ``SourceModel`` contains the information of the earthquake depth, the source type, and the source mechanism. When calculating the Green's function, 
there is no need to specify the source mechanism (but you can provide one anyway), so you can:

.. jupyter-execute::

    source_prem = SourceModel(sdep=16.5, srcType="dc")
    print(source_prem)

``source_prem`` is a ``SourceModel`` that the depth is 16.5km, and the source type is double-couple.

And you will also need to define the velocity and attenuation model. The model contains the information of the thickness, the S wave speed :math:`V_s`, the P wave speed :math:`V_p`, the 
density :math:`\rho`, and the attenuation :math:`Q_s` and :math:`Q_p`. To initialize the ``SeisModel``, we have to prepare a numpy array that stores the model information.
The model is a 2D numpy array following the same format as in ``FK``, and you can directly use ``np.loadtxt`` to load the model file in ``FK``. Each row in the model represents
a layer and the columns are for :math:`thickness` , :math:`V_s` , :math:`V_p` , :math:`\rho` , :math:`Q_s` , :math:`Q_p` . Similar to ``FK``, you can use less than 6 columns. For the detail 
of the model setting, you can refer to ``SeisModel`` in :ref:`:mod:`pyfk.config.config``.

Here we use the PREM model as an example:

.. jupyter-execute::

    from os.path import dirname, join
    from pyfk.tests.taup.test_taup import TestFunctionTaup

    # we use a function in tests to generate the prem model
    prem_data = TestFunctionTaup.gen_test_model("prem")

    prem_data[:5,:]

We can see the top 5 layers of the PREM model. And based on the model data, we can initialize the ``SeisModel``:

.. jupyter-execute::

    model_prem = SeisModel(model=prem_data)
    print(model_prem)

There are also options to control whether we should flatten the model, or whether we are using :math:`\frac{V_p}{V_s}` ratio to get :math:`V_p` in the model data. By default, these two flags are set 
as ``False``.

Based on the ``SourceModel`` and ``SeisModel``, we can now initialize the ``Config`` class. The ``Config`` stores the information such as the receiver depth, the receiver distance, and the 
number of points in the simulation. Generally speaking, they are similar to the flags in ``FK``, but in a more pythonic way.

.. jupyter-execute::

    import numpy as np

    config_prem = Config(
            model=model_prem,
            source=source_prem,
            npt=512,
            dt=0.1,
            receiver_distance=np.arange(10, 40, 10))
    print(config_prem)

For this example, we are using the ``model_prem`` and ``source_prem`` defined previously. And our output should be 512 points with 
0.1s interval. The receiver distances are 10km, 20km, and 30km. If you are planning to use degrees instead, simply set ``degrees=True``,
and the ``receiver_distance`` will be automatically converted to the corresponding distance in km. The default values are set to be the same as
``FK``. One thing to note is that after creation,  ``model_prem``, ``source_prem`` and ``config_prem`` will not be modified by the code, so 
you can reuse it in the future.

Calculate Green's function
---------------------------

After the configuration part, you can simply call ``pyfk.calculate_gf`` to calculate the Green's function.

.. jupyter-execute::

    from pyfk import calculate_gf

    gf = calculate_gf(config_prem)
    print(gf)

``gf`` is a list of ``Stream`` in ``obspy``, the order of ``Stream`` is consistent with the order of ``receiver_distance``.
Each ``Stream`` consists of a list of ``Trace``, and each one represents a component of Green's function. The order of ``Trace`` is
the same as ``FK``. For the meaning of these Green's functions, you might refer to `The introduction from Seisman (In Chinese) <https://blog.seisman.info/fk-notes/>`__.

Each component of the Green's function also contains the header information:

.. jupyter-execute::

    print(gf[0][0].stats)

The header information is the same as the result from ``FK``. The event origin time is at ``1970-01-01T00:00:00.00000Z``, and the starting time
of this ``Trace`` is calculated based on the first arrival time and the sample numbers before the first arrival. For the sac header, ``t1`` is the 
first arrival of P wave and ``t2`` is the first arrival of S wave. ``user1`` and ``user2`` represent the corresponding trace's emission angle. 

If we set ``npt=1`` or ``npt=2`` in the ``Config``, ``calculate_gf`` will not generate ``Stream`` for each distance, but a 1D array.
The 1D array represents the static displacement, similar to ``FK``.

.. jupyter-execute::

    config_prem_static = Config(
            model=model_prem,
            source=source_prem,
            npt=2,
            dt=0.1,
            receiver_distance=np.arange(10, 40, 10))
    gf_static = calculate_gf(config_prem_static)
    print(gf_static[0])

The ``dt`` will be automatically set as 1000 in such the case if the user provided one is smaller than 1000.

We can also have a view of the Green's function:

.. jupyter-execute::

    gf[0][0].plot();

Generate synthetic waveform
---------------------------

By convolving the Green's function with the source time function, we can generate the synthetic waveform. There are mainly
two ways to set up the source time function, similar to ``FK``, that is to generate a trapezoid shaped source time function, or
provide a user defined ``Trace`` containing the source information (only the data attribute is needed at the moment). Note the 
sampling rate of the Green's function and the source time function should be identical.

In this example, we simply call ``pyfk.generate_source_time_function`` to generate a trapezoid shaped source:

.. jupyter-execute::

    from pyfk import generate_source_time_function

    source_time_function=generate_source_time_function(dura=4, rise=0.5, delta=gf[0][0].stats.delta)

And we also need to provide a source mechanism. In ``FK``, the source mechanism is provided as a series of number (-M flag in syn). The source mechanism in 
``pyfk`` should be a list of these numbers. And also you can provide a global CMT solution file:

.. jupyter-execute::

    from pyfk.tests.sync.test_sync import TestFunctionCalculateSync

    test_gcmt_path = TestFunctionCalculateSync.get_sample_gcmt_file_path()
    with open(test_gcmt_path,"r") as f:
        for line in f:
            print(line,end='')

We can read this global CMT solution file, and attach it to ``source_prem``. Something to notice here is that ``source_prem`` is actually 
``config_prem.source``, but ``config_prem.model`` is actually a copy of ``prem_model``.

.. jupyter-execute::

    import obspy
    from pyfk import calculate_sync

    event=obspy.read_events(test_gcmt_path)[0]
    source_prem.update_source_mechanism(event)
    sync_result = calculate_sync(gf, config_prem, 30, source_time_function)

    print(sync_result[0])

The three traces in each Stream is ordered as Z, R, and T components. We can view the synthetic waveform in the Z component:

.. jupyter-execute::

    sync_result[0][0].plot();

It's not strange to have the amplitude around several hundred, as the unit here is cm. But we do see lots of numerical noise. The synthetics 
calculated from ``FK`` will also have this kind of noise. The possible reason might be our largest distance is relative small, thus the upper bound for the integration 
of the wave number will be relative small. 

In ``FK``, it provides the flag to do the waveform integration, the differentiation, and the band-pass filter. Here we can just use ``obspy`` to process the waveform.
For this example, our highest frequency is defined by the Nyquist sampling rate, which is 5HZ. If we want to bandpass the waveform between 0.5HZ and 3HZ, we can:

.. jupyter-execute::

    sync_result_to_filter=sync_result[0][0].copy()
    sync_result_to_filter.detrend("linear")
    sync_result_to_filter.taper(max_percentage=0.05, type='hann')
    sync_result_to_filter.filter("bandpass", freqmin=1/8, freqmax=0.5, corners=2, zerophase=True)
    sync_result_to_filter.plot();

It is the waveform after performing the bandpass filter, with a second stage and two side filter (zerophase=True).

Congratulations! You have finished reading the tutorial part. For more information about the algorithm in ``FK``, you can read the source code in 
:ref:`API Reference`. The package has been tested for several cases using the PREM model with ``FK``, and the the result is close enough. The numerical
difference might be mainly due to that we are using ``double`` but not ``float`` as in ``FK``, or the difference between the programming languages.