pyfk
==========
.. raw:: html

    <a href="https://github.com/ziyixi/pyfk/actions">
        <img src="https://github.com/ziyixi/pyfk/workflows/pyfk/badge.svg"/></a>

.. image:: https://codecov.io/gh/ziyixi/pyfk/branch/master/graph/badge.svg?token=5EL7IDTYLJ
    :target: https://codecov.io/gh/ziyixi/pyfk

.. raw:: html

    <a href="https://ziyixi.github.io/pyfk/">
        <img src="https://img.shields.io/badge/docs-dev-blue.svg"/></a>

.. placeholder-for-doc-index

About
-------------

🚨 **This package is still undergoing rapid development.** 🚨

pyfk is the python version of `FK <http://www.eas.slu.edu/People/LZhu/home.html>`__ used to calculate the Green's function and the synthetic waveforms for the 1D Earth model.

pyfk has mainly provided functions as:

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

pyfk is a free software: you can redistribute it and/or modify it under the terms of
the **MIT License**. A copy of this license is provided in
`LICENSE <https://github.com/ziyixi/pyfk/blob/master/LICENSE>`__.


Acknowledgement
---------------------

The development of pyfk was initially my undergraduate thesis project in USTC, and I would thank Prof. Daoyuan Sun in USTC for mentoring me of this project. I am using
the time of AGU 2020 to refactor my previous work as it's much slower than FK, since I'm simply using numba to speed up the code. And I would also thank my
Ph.D. advisor Prof. Min Chen for supporting my study and research in the US so I can have the time and energy to finish this project.