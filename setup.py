from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(name="random_forest", ext_modules=cythonize('randomforest.pyx'), include_dirs=[numpy.get_include()])