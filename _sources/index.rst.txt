=======================
PyFK
=======================

PyFK is the python port of `FK <http://www.eas.slu.edu/People/LZhu/home.html>`__.

The Frequency-Wavenumber (FK) is a synthetic seismogram package
used to calculate the Green's function and the synthetic waveforms of the 1D Earth model. The first version of FK was developed by Prof. Lupei Zhu in 
1996, and the code was written in Fortran, C, and Perl. 

Nowadays, it's usually more efficient to do seismological research based on a python's workflow, 
with the help of widely used packages such as `Obspy <http://www.eas.slu.edu/People/LZhu/home.html>`__, `Numpy <https://numpy.org/>`__ and `Scipy <https://www.scipy.org/>`__.
Python is also easy to integrate with parallel computing packages either in CPU or GPU to improve the computational speed,


.. include:: ../README.rst
    :start-after: placeholder-for-doc-index

Acknowledgment
---------------------
PyFK was initially my undergraduate thesis project in USTC, so I would thank Prof. Daoyuan Sun in USTC for mentoring me on this project. I would also thank my
Ph.D. advisors Prof. Min Chen and Prof. Songqiao Shawn Wei for supporting my study and research in the United States, so I can have the time and energy to finish this project.

.. toctree::
    :maxdepth: 2
    :hidden:
    :caption: Installing

    introduction/install.rst

.. toctree::
    :maxdepth: 2
    :hidden:
    :caption: Tutorial

    introduction/tutorial.rst

.. toctree::
    :maxdepth: 2
    :hidden:
    :caption: API

    autoapi/index.rst