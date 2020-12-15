"""
Pyfk is the python version of FK used to calculate the Green's function and the synthetic waveforms for the 1D Earth model.
The frequency-Wavenumber (FK) is a synthetic seismogram package used to calculate the Green’s function and the synthetic waveforms of the 1D Earth model. The first version of FK is developed by Prof. Lupei Zhu in 1996, and the code is written in Fortran, C and a Perl interface.
Nowadays, it’s usually efficient to do the seismological research based on a python’s workflow, with the help of widely used packages such as Obspy, Numpy and Scipy. Python is also easy to integrate with parallel computing packages such as mpi4py to do multiple simulations at the same time.
"""
import inspect
import os
import sys

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup

DOCSTRING = __doc__.strip().split("\n")
root_dir = os.path.join(
    os.path.dirname(os.path.abspath(
        inspect.getfile(inspect.currentframe()))), "pyfk"
)

CYTHON_TRACE = 0
if "--CYTHON_TRACE" in sys.argv:
    CYTHON_TRACE = 1
    print("use CYTHON_TRACE")
    sys.argv.remove("--CYTHON_TRACE")

extensions = [
    Extension(
        "pyfk.taup.taup",
        [os.path.join(root_dir, "taup/taup.pyx")],
        include_dirs=[np.get_include()],
        define_macros=[("CYTHON_TRACE", str(CYTHON_TRACE))],
    ),
    Extension(
        "pyfk.gf.waveform_integration",
        [os.path.join(root_dir, "gf/waveform_integration.pyx")],
        include_dirs=[np.get_include()],
        define_macros=[("CYTHON_TRACE", str(CYTHON_TRACE))],
        # extra_compile_args=["-fopenmp"]
    ),
]


def readme():
    with open('README.rst') as f:
        return f.read()


def get_package_data():
    """
    Returns a list of all files needed for the installation relative to the
    'pyfk' subfolder.
    """
    filenames = []
    # Recursively include all files in these folders:
    folders = [
        os.path.join(root_dir, "tests", "data"),
        os.path.join(root_dir, "tests", "data", "hk_gf"),
        os.path.join(root_dir, "tests", "data", "sync_prem_gcmt"),
        os.path.join(root_dir, "tests", "data", "sync_filter"),
        os.path.join(root_dir, "tests", "data", "sync_prem_ep"),
        os.path.join(root_dir, "tests", "data", "sync_prem_sf"),
        os.path.join(root_dir, "tests", "data", "sync_receiver_deeper"),
        os.path.join(root_dir, "tests", "data", "sync_smth"),
    ]
    for folder in folders:
        for directory, _, files in os.walk(folder):
            for filename in files:
                # Exclude hidden files.
                if filename.startswith("."):
                    continue
                filenames.append(
                    os.path.relpath(os.path.join(
                        directory, filename), root_dir)
                )
    return filenames


setup_config = dict(
    name="pyfk",
    version="0.1.0-alpha.1",
    description=DOCSTRING[0],
    long_description=readme(),
    author="Ziyi Xi",
    author_email="xiziyi2015@gmail.com",
    url="https://github.com/ziyixi/pyfk",
    packages=find_packages(),
    license="MIT",
    platforms="OS Independent",
    package_data={"pyfk": get_package_data()},
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    ext_modules=cythonize(extensions, language_level="3", annotate=False),
    zip_safe=False,
    install_requires=["numpy", "obspy", "cython", "scipy", "cysignals"]
)

if __name__ == "__main__":
    setup(
        **setup_config
    )
