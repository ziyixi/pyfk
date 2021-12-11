.. _install:

Installing
==========

Install the PyFK package
-------------------------

The supported Python versions include 3.6, 3.7, and 3.8. Python 2 is not supported. Versions prior (include) 3.5 are not supported as f-string is widely used in the package configuration. Versions after (include) 3.9 are not supported as the Cython compatible issue.

Currently, the package is only supported on Unix-like systems, including Mac and Linux. Windows is not supported as VS compiler doesn't support the C99 standard of "complex.h", which is used in the package. Other platforms such as Mac OS with M1 chip might be supported through installing from PyPi or the source, but it has not been tested.

General Installation
^^^^^^^^^^^^^^^^^^^^^^^^

To install the package on serial mode and the GPU parallel mode, you can install from PyPi::

    pip install pyfk

Anaconda environment is also supported::

    conda install -c ziyixi pyfk

MPI Mode
^^^^^^^^^^^^^^^^^

It's required to compile from the source directly to install the MPI mode.

Firstly we need download the package::

    git clone https://github.com/ziyixi/pyfk
    cd pyfk

Some dependencies should be installed before compiling PyFK::

    conda install -c conda-forge cython numpy obspy scipy cysignals mpi4py

Or from PyPi::

    pip install cython numpy obspy scipy cysignals mpi4py

And now we can compile PyFK as::

    PYFK_USE_MPI=1 python setup.py install

Note that the installation of the mpi4py might need attention if using the cluster. Some clusters might need to specify the MPI lib location to compile the compatible mpi4py.

GPU mode
^^^^^^^^^^^^^^^^^^^^^

Apart from the general Installation, there are some other packages that need to be installed. It's suggested to use Anaconda to download these packages to better incorporate with the CUDA toolkits::

    conda install -c conda-forge cupy numba

Using PyPi is also possible, you can refer to the document of `CuPy <https://docs.cupy.dev/en/stable/install.html>`__ and `Numba <https://numba.pydata.org/numba-doc/latest/user/installing.html#installing-using-pip-on-x86-x86-64-platforms>`__. Note these two packages should be installed with the CUDA support.


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