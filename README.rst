[![FOSSA Status](https://app.fossa.com/api/projects/git%2Bgithub.com%2Fziyixi%2Fpyfk.svg?type=shield)](https://app.fossa.com/projects/git%2Bgithub.com%2Fziyixi%2Fpyfk?ref=badge_shield)

PyFK
==========

.. image:: https://github.com/ziyixi/pyfk/workflows/pyfk/badge.svg
    :target: https://github.com/ziyixi/pyfk/actions

.. image:: https://codecov.io/gh/ziyixi/pyfk/branch/master/graph/badge.svg?token=5EL7IDTYLJ
    :target: https://codecov.io/gh/ziyixi/pyfk

.. image:: https://img.shields.io/badge/docs-dev-blue.svg
    :target: https://ziyixi.github.io/pyfk/

.. image:: https://badge.fury.io/py/pyfk.svg
    :target: https://badge.fury.io/py/pyfk

.. image:: https://anaconda.org/ziyixi/pyfk/badges/version.svg
    :target: https://anaconda.org/ziyixi/pyfk

.. image:: https://anaconda.org/ziyixi/pyfk/badges/platforms.svg
    :target: https://github.com/ziyixi/pyfk

.. image:: https://anaconda.org/ziyixi/pyfk/badges/license.svg
    :target: https://github.com/ziyixi/pyfk/blob/master/LICENSE

.. placeholder-for-doc-index

About
-------------

PyFK is the python port of `FK <http://www.eas.slu.edu/People/LZhu/home.html>`__ used to calculate the Green's function and the synthetic waveforms for the 1D Earth model.

The main features of this package are:

* Compute the Green's function for the explosion, single force, and double couple source using the frequency-wavenumber method.
* Compute the static displacements and corresponding Green's function.
* Compute the synthetic waveforms by convolving Green's function with the seismic source.
* Use the seismic data format of Obspy, which is easy to perform the signal processing.

And the package is unique as:

* all the code is written in pure python, and it's compatible with Unix-like systems including Mac and Linux. The Windows is not supported, as the package uses the complex number in Cython, which uses the C99 standard of "complex.h" that has not been supported by the Visual Studio compiler.
* it uses Cython to speed up the computationally expensive part (mainly the wavenumber integration).
* The package has also provided three modes:
  
    * Serial mode: the serial version implements the FK algorithm in Python.
    * Parallel mode on CPU: the wavenumber integration can be paralleled by MPI. 
    * Parallel mode on GPU: the wavenumber integration can also be paralleled by CUDA on GPU.

Installation
-------------

The serial version and the parallel version on GPU can be simply installed using pip:

.. code-block:: bash

    pip install pyfk

Or conda::

    conda install -c ziyixi pyfk

Extra packages will be required to enable the GPU mode. For the MPI mode, it's suggested to directly compile from the source. For more details about the Installation, you can refer to the Installing part of the document.

Todo
------------------
The current bugs to fix or features to provide are listed in `TODO <https://github.com/ziyixi/pyfk/issues/5>`__.

License
-------

PyFK is a free software: you can redistribute it and modify it under the terms of
the **MIT License**. A copy of this license is provided in
`LICENSE <https://github.com/ziyixi/pyfk/blob/master/LICENSE>`__.


[![FOSSA Status](https://app.fossa.com/api/projects/git%2Bgithub.com%2Fziyixi%2Fpyfk.svg?type=large)](https://app.fossa.com/projects/git%2Bgithub.com%2Fziyixi%2Fpyfk?ref=badge_large)