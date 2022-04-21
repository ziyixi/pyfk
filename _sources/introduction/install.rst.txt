.. _install:

Installing
==========

Install the PyFK package
-------------------------

The supported Python versions include 3.7, 3.8, and 3.9. Python 2 is not supported. Versions prior (include) 3.6 are not supported. Versions after (include) 3.10 are not supported as the Cython compatible issue.

Currently, the package is only supported on Unix-like systems, including Mac and Linux. Windows is not supported as VS compiler doesn't support the C99 standard of "complex.h", which is used in the package. Other platforms such as Mac OS with M1 chip might be supported through installing from PyPi or the source, but it has not been tested.

General Installation
^^^^^^^^^^^^^^^^^^^^^^^^

To install the package on serial mode and the GPU parallel mode, you can install from PyPi::

    pip install pyfk

Note this command might fail in some rare cases if mpi4py can not be installed in the local system. You may have to install openmpi or mpich, or try::
    
    pip install poetry-core@https://github.com/python-poetry/poetry-core/archive/refs/tags/1.1.0a7.zip
    # or any version later than 1.1.0a7, or just poetry-core when poetry 1.1.0 has been released
    pip install numpy obspy cython scipy cysignals
    pip install pyfk --no-build-isolation

Anaconda environment is also supported::

    conda install -c ziyixi pyfk

MPI Mode
^^^^^^^^^^^^^^^^^

To install the package with the MPI mode, you can install it from PyPi::

    PYFK_USE_MPI=1 pip install pyfk[mpi]

:code:`PYFK_USE_MPI` is a environment variable that should be set to 1 to enable the MPI mode. This environment is only required in installation. The package with the MPI supporting can fully replace the package in normal mode by providing extra features in running the code in parallel (with mpirun).

Note a working MPI C compile is required for compilation. Conda is not supported for MPI mode, as the MPI is system dependent, especially for the case in the super computers.

If you have installed mpi4py using conda before, it is advised to reinstall it using PyPi or uninstall it. Conda will attempt to install its own version of MPI, which usually works fine for a single node. But in the case of super computers, it might conflict with commands like "srun" or "ibrun".

GPU mode
^^^^^^^^^^^^^^^^^^^^^

Apart from the general Installation, there are some other packages that need to be installed. It's suggested to use Anaconda to download these packages to better incorporate with the CUDA toolkits::

    conda install -c conda-forge cupy numba

Special note to check the GPU driver version and the CUDA version using::

    nvidia-smi

As the version for cudatoolkit should be equal or smaller than the driver's supported version. Higher versions might have the risk of the CUDA runtime error. For example, on the local cluster on MSU ICER, the supported CUDA version is 11.5, while using the above conda command will install the CUDA 11.6 toolkit at the time of April 2022. So instead, for this case, we can use::

    conda install -c conda-forge cupy numba cudatoolkit=11.5

Using PyPi to install these two packages is also possible, you can refer to the document of `CuPy <https://docs.cupy.dev/en/stable/install.html>`__ and `Numba <https://numba.pydata.org/numba-doc/latest/user/installing.html#installing-using-pip-on-x86-x86-64-platforms>`__. Note these two packages should be installed with the CUDA support.

These two packages can be installed before or after installation PyFK, either in the normal mode or the MPI mode. There is nothing special for installing PyFK in the GPU mode.

Test the Installation
---------------------------

To test the installation of the package, you can import the package::

    import pyfk
    pyfk.mpi_info()

If MPI mode is enabled, it will output the MPI related information. If not, it will output "MPI is not used.".

To have a complete test of PyFK, several other packages should be downloaded::

    conda install pytest pytest-mpi

Or::

    pip install pytest pytest-mpi

After installing the packages, you can run the following command::

    pytest --pyargs pyfk

Several warnings can be omitted, such as dividing a zero, which is only related to cross-correlate two waveforms with all values as 0.

To test the GPU mode, you can run::

    PYFK_USE_CUDA=1 pytest --pyargs pyfk

To test the MPU version, you can run::

    mpirun -np 3 pytest --with-mpi --pyargs pyfk

for three processes' parallation.