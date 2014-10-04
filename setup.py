# -*- coding: utf-8 -*-
from distutils.core import setup
from Cython.Build import cythonize

setup(
    name = 'cdetector',
    ext_modules = cythonize("cdetector.pyx"),
)