.. _install:

Installing
==========

The environments
--------------------

The package pyfk runs on Unix-like systems including Mac and Linux and you will need **Python 3.6 or greater**. Python 3.9 is not supported yet as it's having problems to run Cython at the moment.

At the current stage, the only way to install the package is directly from the Github repo, and it's recommended to use anaconda as your python environment::

    conda install numpy obspy cython scipy
    conda install -c conda-forge cysignals
    pip install git+https://github.com/ziyixi/pyfk

You can also use pipenv to install the package::

    curl -OL https://raw.githubusercontent.com/ziyixi/pyfk/master/Pipfile
    pipenv [--python=<your desired python interpreter (if omitted, will use the default one)>] install
    pipenv install -e git+https://github.com/ziyixi/pyfk#egg=pyfk

Or directly use pip::

    pip install numpy obspy cython scipy cysignals
    pip install git+https://github.com/ziyixi/pyfk

This will allow you to use pyfk from python.


Test your installing
--------------------

Pyfk ships with a full test suite. You can run the tests after you install it but you will need a few extra dependencies as well, if you are using pipenv::

    pipenv install pytest

Or conda::

    conda install pytest

Or pip::

    pip install pytest

And now you can run the tests (remember to activate the virtual environment such as ``pipenv shell`` or ``conda activate <your virtual env name>`` first)::

    pytest --pyargs pyfk

You may ignore the warnings as they are not essentially important. 