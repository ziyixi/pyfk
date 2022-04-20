"""
Pyfk is the python version of FK used to calculate the Green's function and the synthetic waveforms for the 1D Earth model.
The frequency-Wavenumber (FK) is a synthetic seismogram package used to calculate the Green’s function and the synthetic waveforms of the 1D Earth model. The first version of FK is developed by Prof. Lupei Zhu in 1996, and the code is written in Fortran, C and a Perl interface.
Nowadays, it’s usually efficient to do the seismological research based on a python’s workflow, with the help of widely used packages such as Obspy, Numpy and Scipy. Python is also easy to integrate with parallel computing packages such as mpi4py to do multiple simulations at the same time.
"""
import inspect
import os

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension


def build(setup_kwargs):
    # * root dir
    root_dir = os.path.join(
        os.path.dirname(os.path.abspath(
            inspect.getfile(inspect.currentframe()))), "pyfk"
    )
    # * MPI mode
    compile_time_env = {
        "PYFK_USE_MPI": "0"
    }
    PYFK_USE_MPI = os.getenv("PYFK_USE_MPI", "0")
    mpi_link_args = []
    mpi_include_dirs = [np.get_include()]
    if PYFK_USE_MPI == "1":
        os.environ["CC"] = "mpicc"
        compile_time_env["PYFK_USE_MPI"] = "1"
        mpi_link_args.append("-lmpi")
        try:
            import mpi4py
        except:
            raise Exception(
                "please install mpi4py first to enable the MPI mode!")
        mpi_include_dirs.append(mpi4py.get_include())

    # * cysignals
    def get_include_cysignals():
        import cysignals
        return os.path.join(os.path.dirname(cysignals.__file__), 'include')
    mpi_include_dirs.append(get_include_cysignals())

    # * only for debug purpose
    # ref to https://cython.readthedocs.io/en/latest/src/tutorial/profiling_tutorial.html#enabling-line-tracing
    CYTHON_TRACE = 0
    PYFK_USE_CYTHON_TRACE = os.getenv("PYFK_USE_CYTHON_TRACE", "0")
    if PYFK_USE_CYTHON_TRACE == "1":
        CYTHON_TRACE = 1

    # * extensions
    extensions = [
        Extension(
            "pyfk.taup.taup",
            [os.path.join(root_dir, "taup/taup.pyx")],
            include_dirs=[np.get_include()],
            define_macros=[("CYTHON_TRACE", str(CYTHON_TRACE))],
            language="c"
        ),
        Extension(
            "pyfk.gf.waveform_integration",
            [os.path.join(root_dir, "gf/waveform_integration.pyx")],
            include_dirs=mpi_include_dirs,
            define_macros=[("CYTHON_TRACE", str(CYTHON_TRACE))],
            language="c",
            extra_link_args=mpi_link_args
        ),
    ]
    # * update setup
    setup_kwargs.update(
        dict(
            ext_modules=cythonize(extensions, language_level=3,
                                  annotate=False, compile_time_env=compile_time_env),
            zip_safe=False
        )
    )
