pyfk
==========

.. image:: https://github.com/ziyixi/pyfk/workflows/pyfk/badge.svg
    :target: https://github.com/ziyixi/pyfk/actions

.. image:: https://codecov.io/gh/ziyixi/pyfk/branch/master/graph/badge.svg?token=5EL7IDTYLJ
    :target: https://codecov.io/gh/ziyixi/pyfk

.. image:: https://img.shields.io/badge/docs-dev-blue.svg
    :target: https://ziyixi.github.io/pyfk/

.. image:: https://badge.fury.io/py/pyfk.svg
    :target: https://badge.fury.io/py/pyfk

.. placeholder-for-doc-index

About
-------------

🚨 **This package is still undergoing rapid development.** 🚨

Pyfk is the python version of `FK <http://www.eas.slu.edu/People/LZhu/home.html>`__ used to calculate the Green's function and the synthetic waveforms for the 1D Earth model.

Pyfk has mainly provided functions as:

* compute the Green's function for the explosion, single force, and double couple source using the frequency-wavenumber method.
* compute the static displacements and corresponding Green's function.
* compute the synthetic waveforms by convolving the Green's function with the seismic source.
* have a close integration with Obspy, and is naturally to process the waveforms using this package.

and it has the features as:

* all the code is written in pure python, and it's compatible with Linux and Mac at the moment. The Windows is not supported, as i am 
  using the complex number in Cython, which will use the C99 standard of "complex.h" that has not been supported by the Visual Studio compiler.
* it uses Cython to speed up the computationally expensive part (mainly the wavenumber integration).
* a complete test has been performed to ensure pyfk has the same result as FK.


License
-------

Pyfk is a free software: you can redistribute it and/or modify it under the terms of
the **MIT License**. A copy of this license is provided in
`LICENSE <https://github.com/ziyixi/pyfk/blob/master/LICENSE>`__.
