# -*- coding: utf-8 -*-
from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name = 'cycounting',
    ext_modules = cythonize("cycounting.pyx"),
    include_dirs = [np.get_include()]
)