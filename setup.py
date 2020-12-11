"""
pyfk
"""
import inspect
import os
import sys

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup

DOCSTRING = __doc__.strip().split("\n")
root_dir = os.path.join(
    os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))), "pyfk"
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
        # extra_compile_args=["-fsanitize=address"]
    ),
]


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
    ]
    for folder in folders:
        for directory, _, files in os.walk(folder):
            for filename in files:
                # Exclude hidden files.
                if filename.startswith("."):
                    continue
                filenames.append(
                    os.path.relpath(os.path.join(directory, filename), root_dir)
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
    zip_safe=False,
)

if __name__ == "__main__":
    setup(
        extras_require={
            "dev": [
                "appdirs==1.4.4",
                "attrs==20.3.0; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3'",
                "autopep8==1.5.4",
                "black==19.10b0; python_version >= '3.6'",
                "cached-property==1.5.2",
                "cerberus==1.3.2",
                "certifi==2020.12.5",
                "chardet==3.0.4",
                "click==7.1.2; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3, 3.4'",
                "colorama==0.4.4; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3, 3.4'",
                "coverage==5.3",
                "distlib==0.3.1",
                "idna==2.10; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3'",
                "importlib-metadata==3.1.1; python_version < '3.8'",
                "iniconfig==1.1.1",
                "orderedmultidict==1.0.1",
                "packaging==20.7; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3'",
                "pathspec==0.8.1",
                "pep517==0.9.1",
                "pip-shims==0.5.3; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3, 3.4'",
                "pipenv-setup==3.1.1",
                "pipfile==0.0.2",
                "plette[validation]==0.2.3; python_version >= '2.6' and python_version not in '3.0, 3.1, 3.2, 3.3'",
                "pluggy==0.13.1; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3'",
                "py==1.9.0; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3'",
                "pycodestyle==2.6.0; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3'",
                "pyparsing==2.4.7; python_version >= '2.6' and python_version not in '3.0, 3.1, 3.2, 3.3'",
                "pytest==6.1.2",
                "python-dateutil==2.8.1; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3'",
                "regex==2020.11.13",
                "requests==2.25.0; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3, 3.4'",
                "requirementslib==1.5.16; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3, 3.4'",
                "six==1.15.0; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3'",
                "toml==0.10.2; python_version >= '2.6' and python_version not in '3.0, 3.1, 3.2, 3.3'",
                "tomlkit==0.7.0; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3, 3.4'",
                "typed-ast==1.4.1; python_version < '3.8' and implementation_name == 'cpython'",
                "urllib3==1.26.2; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3, 3.4' and python_version < '4'",
                "vistir==0.5.2; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3'",
                "wheel==0.36.1; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3, 3.4'",
                "zipp==3.4.0; python_version < '3.8'",
            ]
        },
        install_requires=[
            "certifi==2020.12.5",
            "chardet==3.0.4",
            "cycler==0.10.0",
            "cysignals==1.10.2",
            "cython==0.29.21",
            "decorator==4.4.2",
            "future==0.18.2; python_version >= '2.6' and python_version not in '3.0, 3.1, 3.2, 3.3'",
            "idna==2.10; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3'",
            "kiwisolver==1.3.1; python_version >= '3.6'",
            "lxml==4.6.2; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3, 3.4'",
            "matplotlib==3.3.3; python_version >= '3.6'",
            "numpy==1.19.4",
            "obspy==1.2.2",
            "pillow==8.0.1; python_version >= '3.6'",
            "pyparsing==2.4.7; python_version >= '2.6' and python_version not in '3.0, 3.1, 3.2, 3.3'",
            "python-dateutil==2.8.1; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3'",
            "requests==2.25.0; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3, 3.4'",
            "scipy==1.5.4",
            "six==1.15.0; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3'",
            "sqlalchemy==1.3.20; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3'",
            "urllib3==1.26.2; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3, 3.4' and python_version < '4'",
        ],
        **setup_config
    )
