.. _parallel:

Parallelization
==========

Why are we using ``PyFK``? You may think it's more pythonic and can tolerant the possible inefficiency. However, with the parallelization, it can be much faster than you may think.

The idea to parallelize the FK algorithm is simple, based on `The introduction from Seisman (In Chinese) <https://blog.seisman.info/fk-notes/>`__, we know the simulate consists of two levels of the loop, one is for the frequency, and another is for the wavenumber. So these two loops can be flattened to one large loop and apply parallelism. 

The parallelization is designed only to calculate the Green's function as it's the most time-consuming part. The calculation of the synthetic waveforms is only about convolving the source, which is much faster.

MPI Mode
-------------------

To use the MPI mode, you have to compile from the source with the environment ``PYFK_USE_MPI`` set as 1. The reason for this design is that the MPI mode relies on ``mpi4py``, which has the dependency of the system-provided MPI package. Everything should be compiled in the local machine, not to mention we are using MPI inside Cython. 

After successfully installing ``PyFK`` with the MPI support. You can directly use it to perform the simulation. Assume we have a Python script::

    import numpy as np
    import obspy
    from pyfk.config.config import Config, SeisModel, SourceModel
    from pyfk.gf.gf import calculate_gf
    from pyfk.tests.taup.test_taup import TestFunctionTaup

    model_data = SeisModel(model=TestFunctionTaup.gen_test_model("prem"))
    source = SourceModel(sdep=15)
        config = Config(
        model=model,
        source=source,
        npt=512,
        dt=0.1,
        receiver_distance=[
            10,
            20,
            30])

    result = calculate_gf(config)

This script loads the PREM model and calculates Green's function based on our setting. It is the same as the serial mode, but when ``calculate_gf`` detects it's called by ``mpirun``, it will automatically be run with MPI. The calculated Green's function is broadcasted to all the processes, and can be further processed by only considering the main process such as setting ``rank==0``. I will not talk about this part which should be introduced in `The document of mpi4py <https://mpi4py.readthedocs.io/en/stable/>`__. 


GPU mode
-----------------

The GPU mode can be enabled by passing ``cuda=True`` in ``Config``, or set the system environment ``PYFK_USE_CUDA`` as 1. ``PYFK_USE_CUDA`` will only be read during the calculation, which has no relationship with the package installation part. Either ``cuda=True`` or ``PYFK_USE_CUDA`` will enable the GPU mode.

To enable the GPU mode, additional packages are required, including Numba and CuPy. Note these packages are not automatically installed by ``pip`` or ``conda`` to avoid the unnecessary package dependencies. Numba is used to compile the CUDA kernel, while CuPy provides additional functions to calculate the special functions on GPU, which is used in this package.

For future versions, these dependencies might be removed by directly using CUDA.

For consistency, you don't have to make any additional modifications to the existing code. As long as the GPU mode is enabled, the calculation of Green's function will be automatically on GPU. It's suggested to perform benchmark before migrating to GPU, as smaller jobs might not be worthy.

There might have a problem that the GPU memory is not big enough. It's possible as we first generate all the information before performing the two-loop integration, then move all the info to GPU. A workaround to this problem is to divide the large memory information into small subsets; after one subset has finished calculation, we move another one. ``PyFK`` has provided a system environment variable named ``CUDA_DIVIDE_NUM``. If this environment has not been set, its value will be one, which means there will only be one subset. Try to make this value larger when there is not enough GPU memory.

The bottleneck of the GPU calculation is the data moving. Currently, there is no better way to solve this problem; using thread is not the way to go as the actual calculation on GPU is pretty fast. So it's not surprising if you look at the system monitor, and the GPU efficiency is only 0%, although we know it's running on GPU and it's fast. Look at the GPU memory usage and you will get it that we are using GPU.