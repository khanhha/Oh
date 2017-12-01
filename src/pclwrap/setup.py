import sys
import numpy
from distutils.core import setup, Extension
from Cython.Build import cythonize

compile_args = []

if sys.platform == 'darwin':
    compile_args.append('-mmacosx-version-min=10.7')

module = Extension('pcl._pcl',
                sources=['_pcl.pyx'],
                include_dirs= ['..//..//src', '..//..//3rdParty//eigen', '..//..//3rdParty', numpy.get_include()],
                extra_compile_args=compile_args,
                language='c++')

ext_module = cythonize(module)
setup(
    name='pcl',
    packages=['pcl'],
    ext_modules= ext_module
)

# # -*- coding: utf-8 -*-
# from __future__ import print_function
# from collections import defaultdict
# from Cython.Distutils import build_ext
# from distutils.core import setup
# from distutils.extension import Extension
# # from Cython.Build import cythonize    # MacOS NG
# from setuptools import setup, find_packages, Extension
#
# import subprocess
# import numpy
# import sys
# import platform
# import os
# import time
#
#
# setup_requires = []
# install_requires = [
#     'filelock',
#     'nose',
#     'numpy',
#     'Cython>=0.25.2',
# ]
#
# compile_args = []
#
# if sys.platform == 'darwin':
#     compile_args.append('-mmacosx-version-min=10.7')
#
# ext_args = defaultdict(list)
#
# ext_args['extra_compile_args'].append(compile_args)
#
# # set include path
# ext_args['include_dirs'].append(numpy.get_include())
# ext_args['include_dirs'].append('G:\\Projects\\Oh\Oh\\3rdParty\eigen')
# ext_args['include_dirs'].append('G:\\Projects\\Oh\\Oh\\src')
#
# module = [Extension("pcl._pcl", ["_pcl.pyx"], language="c++", **ext_args)]
#
# setup(name='python-pcl',
#       description='pcl wrapper',
#       url='http://github.com/strawlab/python-pcl',
#       version='0.2',
#       author='John Stowers',
#       author_email='john.stowers@gmail.com',
#       license='BSD',
#       packages=["pcl"],
#       zip_safe=False,
#       setup_requires=setup_requires,
#       install_requires=install_requires,
#       ext_modules=module
#     )
