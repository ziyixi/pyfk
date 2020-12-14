=======================
pyfk
=======================

The package pyfk is the python version of `FK <http://www.eas.slu.edu/People/LZhu/home.html>`__.

The frequency-Wavenumber (FK) is a synthetic seismogram package
used to calculate the Green's function and the synthetic waveforms for the 1D Earth model. The first version of FK is developed by Prof. Lupei Zhu in 
1996, and the code is written in Fortran, C and a Perl interface. 

Nowadays, it's usually efficient to do the seismological research based on a python's workflow, 
with the help of widely used packages such as `Obspy <http://www.eas.slu.edu/People/LZhu/home.html>`__, `Numpy <https://numpy.org/>`__ and `Scipy <https://www.scipy.org/>`__.
It's also easy to integrate with parallel computing packages such as `mpi4py <https://github.com/mpi4py/mpi4py>`__ to do multiple simulations at the same time.


.. include:: ../README.rst
    :start-after: placeholder-for-doc-index

.. toctree::
    :maxdepth: 2
    :hidden:
    :caption: API

    autoapi/index.rst