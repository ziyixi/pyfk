from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy as np

sourcefiles = ["pyfk/taup/taup.pyx"]

extensions = [Extension("pyfk", sourcefiles, include_dirs=[
                        np.get_include()])]

setup(
    ext_modules=cythonize(extensions, language_level="3", annotate=True),
    zip_safe=False,
)
