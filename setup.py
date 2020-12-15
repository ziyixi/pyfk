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
        # extra_compile_args=["-fopenmp"]
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
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    ext_modules=cythonize(extensions, language_level="3", annotate=False),
    zip_safe=False,
)

if __name__ == "__main__":
    setup(
        extras_require={
            "dev": [
                "alabaster==0.7.12",
                "appdirs==1.4.4",
                "appnope==0.1.2; sys_platform == 'darwin' and platform_system == 'Darwin'",
                "argon2-cffi==20.1.0",
                "astroid==2.4.2; python_version >= '3'",
                "async-generator==1.10; python_version >= '3.5'",
                "attrs==20.3.0; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3'",
                "autopep8==1.5.4",
                "babel==2.9.0; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3'",
                "backcall==0.2.0",
                "black==19.10b0; python_version >= '3.6'",
                "bleach==3.2.1; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3, 3.4'",
                "cached-property==1.5.2",
                "cerberus==1.3.2",
                "certifi==2020.12.5",
                "cffi==1.14.4",
                "chardet==3.0.4",
                "click==7.1.2; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3, 3.4'",
                "colorama==0.4.4; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3, 3.4'",
                "coverage==5.3",
                "decorator==4.4.2",
                "defusedxml==0.6.0; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3, 3.4'",
                "distlib==0.3.1",
                "docutils==0.16",
                "entrypoints==0.3; python_version >= '2.7'",
                "idna==2.10; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3'",
                "imagesize==1.2.0; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3'",
                "iniconfig==1.1.1",
                "ipykernel==5.4.2; python_version >= '3.5'",
                "ipython==7.19.0; python_version >= '3.3'",
                "ipython-genutils==0.2.0",
                "ipywidgets==7.5.1",
                "jedi==0.17.2; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3, 3.4'",
                "jinja2==2.11.2; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3, 3.4'",
                "jsonschema==3.2.0",
                "jupyter-client==6.1.7; python_version >= '3.5'",
                "jupyter-core==4.7.0; python_version >= '3.6'",
                "jupyter-sphinx==0.3.2",
                "jupyterlab-pygments==0.1.2",
                "lazy-object-proxy==1.4.3; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3'",
                "markupsafe==1.1.1; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3'",
                "mistune==0.8.4",
                "nbclient==0.5.1; python_version >= '3.6'",
                "nbconvert==6.0.7; python_version >= '3.6'",
                "nbformat==5.0.8; python_version >= '3.5'",
                "nest-asyncio==1.4.3; python_version >= '3.5'",
                "notebook==6.1.5; python_version >= '3.5'",
                "orderedmultidict==1.0.1",
                "packaging==20.8; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3'",
                "pandocfilters==1.4.3",
                "parso==0.7.1; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3'",
                "pathspec==0.8.1",
                "pep517==0.9.1",
                "pexpect==4.8.0; sys_platform != 'win32'",
                "pickleshare==0.7.5",
                "pip-shims==0.5.3; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3, 3.4'",
                "pipenv-setup==3.1.1",
                "pipfile==0.0.2",
                "plette[validation]==0.2.3; python_version >= '2.6' and python_version not in '3.0, 3.1, 3.2, 3.3'",
                "pluggy==0.13.1; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3'",
                "prometheus-client==0.9.0",
                "prompt-toolkit==3.0.8; python_full_version >= '3.6.1'",
                "ptyprocess==0.6.0; os_name != 'nt'",
                "py==1.10.0; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3'",
                "pycodestyle==2.6.0; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3'",
                "pycparser==2.20; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3'",
                "pydata-sphinx-theme==0.4.1",
                "pygments==2.7.3; python_version >= '3.5'",
                "pyparsing==2.4.7; python_version >= '2.6' and python_version not in '3.0, 3.1, 3.2, 3.3'",
                "pyrsistent==0.17.3; python_version >= '3.5'",
                "pytest==6.2.0",
                "python-dateutil==2.8.1; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3'",
                "pytz==2020.4",
                "pyyaml==5.3.1",
                "pyzmq==20.0.0; python_version >= '3.5'",
                "regex==2020.11.13",
                "requests==2.25.0; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3, 3.4'",
                "requirementslib==1.5.16; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3, 3.4'",
                "rstcheck==3.3.1",
                "send2trash==1.5.0",
                "six==1.15.0; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3'",
                "snowballstemmer==2.0.0",
                "sphinx==3.3.1",
                "sphinx-autoapi==1.5.1",
                "sphinxcontrib-applehelp==1.0.2; python_version >= '3.5'",
                "sphinxcontrib-devhelp==1.0.2; python_version >= '3.5'",
                "sphinxcontrib-htmlhelp==1.0.3; python_version >= '3.5'",
                "sphinxcontrib-jsmath==1.0.1; python_version >= '3.5'",
                "sphinxcontrib-qthelp==1.0.3; python_version >= '3.5'",
                "sphinxcontrib-serializinghtml==1.1.4; python_version >= '3.5'",
                "terminado==0.9.1; python_version >= '3.6'",
                "testpath==0.4.4",
                "toml==0.10.2; python_version >= '2.6' and python_version not in '3.0, 3.1, 3.2, 3.3'",
                "tomlkit==0.7.0; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3, 3.4'",
                "tornado==6.1; python_version >= '3.5'",
                "traitlets==5.0.5; python_version >= '3.7'",
                "typed-ast==1.4.1; python_version < '3.8' and implementation_name == 'cpython'",
                "unidecode==1.1.1",
                "urllib3==1.26.2; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3, 3.4' and python_version < '4'",
                "vistir==0.5.2; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3'",
                "wcwidth==0.2.5",
                "webencodings==0.5.1",
                "wheel==0.36.2; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3, 3.4'",
                "widgetsnbextension==3.5.1",
                "wrapt==1.12.1",
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
