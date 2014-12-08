# -*- coding: utf-8 -*-
from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name = 'cdetector',
    ext_modules = cythonize("cdetector.pyx"),
    include_dirs = [np.get_include()]
)