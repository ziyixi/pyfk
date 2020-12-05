"""
"""
import inspect
import os

from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy as np
from setuptools.command.install import install
from setuptools.command.develop import develop
import sys

DOCSTRING = __doc__.strip().split("\n")
root_dir = os.path.join(
    os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))),
    "pyfk"
)

CYTHON_TRACE = 0
if "--CYTHON_TRACE" in sys.argv:
    CYTHON_TRACE = 1
    print("use CYTHON_TRACE")
    sys.argv.remove("--CYTHON_TRACE")

extensions = [
    Extension("pyfk.taup.taup", [os.path.join(root_dir, "taup/taup.pyx")],
              include_dirs=[np.get_include()], define_macros=[("CYTHON_TRACE", str(CYTHON_TRACE))])
]


def get_package_data():
    """
    Returns a list of all files needed for the installation relative to the
    'pyfk' subfolder.
    """
    filenames = []
    # The lasif root dir.
    root_dir = os.path.join(
        os.path.dirname(os.path.abspath(
            inspect.getfile(inspect.currentframe()))),
        "pyfk",
    )
    # Recursively include all files in these folders:
    folders = [os.path.join(root_dir, "tests", "data")]
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
    version="0.0.1",
    description=DOCSTRING[0],
    long_description="\n".join(DOCSTRING),
    author="Ziyi Xi",
    author_email="xiziyi2015@gmail.com",
    url="https://github.com/ziyixi/pyfk",
    packages=find_packages(),
    license="MIT",
    platforms="OS Independent",
    package_data={"pyfk": get_package_data()},
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 1 - Planning",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6" "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    ext_modules=cythonize(extensions, language_level="3", annotate=True),
    zip_safe=False
)


if __name__ == "__main__":
    setup(
        extras_require={
            "dev": [
                "appdirs==1.4.4",
                "appnope==0.1.2; sys_platform == 'darwin'",
                "attrs==20.3.0; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3'",
                "autopep8==1.5.4",
                "backcall==0.2.0",
                "black==19.10b0; python_version >= '3.6'",
                "cached-property==1.5.2",
                "cerberus==1.3.2",
                "certifi==2020.11.8",
                "chardet==3.0.4",
                "click==7.1.2; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3, 3.4'",
                "colorama==0.4.4; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3, 3.4'",
                "decorator==4.4.2",
                "distlib==0.3.1",
                "idna==2.10; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3'",
                "ipython==7.19.0",
                "ipython-genutils==0.2.0",
                "jedi==0.17.2; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3, 3.4'",
                "llvmlite==0.35.0; python_version >= '3.6'",
                "numba==0.52.0",
                "numpy==1.19.4",
                "orderedmultidict==1.0.1",
                "packaging==20.7; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3'",
                "parso==0.7.1; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3'",
                "pathspec==0.8.1",
                "pep517==0.9.1",
                "pexpect==4.8.0; sys_platform != 'win32'",
                "pickleshare==0.7.5",
                "pip-shims==0.5.3; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3, 3.4'",
                "pipenv-setup==3.1.1",
                "pipfile==0.0.2",
                "plette[validation]==0.2.3; python_version >= '2.6' and python_version not in '3.0, 3.1, 3.2, 3.3'",
                "prompt-toolkit==3.0.8; python_full_version >= '3.6.1'",
                "ptyprocess==0.6.0",
                "pycodestyle==2.6.0; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3'",
                "pygments==2.7.2; python_version >= '3.5'",
                "pyparsing==2.4.7; python_version >= '2.6' and python_version not in '3.0, 3.1, 3.2, 3.3'",
                "python-dateutil==2.8.1; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3'",
                "regex==2020.11.13",
                "requests==2.25.0; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3, 3.4'",
                "requirementslib==1.5.16; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3, 3.4'",
                "six==1.15.0; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3'",
                "toml==0.10.2; python_version >= '2.6' and python_version not in '3.0, 3.1, 3.2, 3.3'",
                "tomlkit==0.7.0; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3, 3.4'",
                "traitlets==5.0.5; python_version >= '3.7'",
                "typed-ast==1.4.1",
                "urllib3==1.26.2; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3, 3.4' and python_version < '4'",
                "vistir==0.5.2; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3'",
                "wcwidth==0.2.5",
                "wheel==0.36.1; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3, 3.4'",
            ]
        },
        install_requires=[
            "attrs==20.3.0; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3'",
            "autopep8==1.5.4",
            "certifi==2020.11.8",
            "chardet==3.0.4",
            "coverage==5.3",
            "cycler==0.10.0",
            "cython==0.29.21",
            "decorator==4.4.2",
            "future==0.18.2; python_version >= '2.6' and python_version not in '3.0, 3.1, 3.2, 3.3'",
            "idna==2.10; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3'",
            "iniconfig==1.1.1",
            "kiwisolver==1.3.1; python_version >= '3.6'",
            "lxml==4.6.2; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3, 3.4'",
            "matplotlib==3.3.3; python_version >= '3.6'",
            "numpy==1.19.4",
            "obspy==1.2.2",
            "packaging==20.7; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3'",
            "pillow==8.0.1; python_version >= '3.6'",
            "pluggy==0.13.1; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3'",
            "py==1.9.0; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3'",
            "pycodestyle==2.6.0; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3'",
            "pyparsing==2.4.7; python_version >= '2.6' and python_version not in '3.0, 3.1, 3.2, 3.3'",
            "pytest==6.1.2",
            "python-dateutil==2.8.1; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3'",
            "requests==2.25.0; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3, 3.4'",
            "scipy==1.5.4; python_version >= '3.6'",
            "six==1.15.0; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3'",
            "sqlalchemy==1.3.20; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3'",
            "toml==0.10.2; python_version >= '2.6' and python_version not in '3.0, 3.1, 3.2, 3.3'",
            "urllib3==1.26.2; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3, 3.4' and python_version < '4'",
        ],
        **setup_config
    )
