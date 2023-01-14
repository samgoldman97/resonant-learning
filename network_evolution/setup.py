#!/usr/bin/env python

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy


# Compile with:
# $ python3 setup.py build_ext --inplace
setup(
    cmdclass={"build_ext": build_ext},
    ext_modules=[
        Extension(
            "scoring",
            sources=["scoring_function.pyx", "scoring.c", "dict.c", "helpers.c"],
            include_dirs=[numpy.get_include()],
            extra_compile_args=["-O3"],
        )
    ],
)
