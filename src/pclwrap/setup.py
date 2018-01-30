import sys
import numpy
import glob, os
from distutils.core import setup, Extension
from Cython.Build import cythonize

compile_args = []

if sys.platform == 'darwin':
    compile_args.append('-mmacosx-version-min=10.7')

#clear file
for root, dirs, files in os.walk("..//..//src"):
    for file in files:
        if file == '_pcl.cpp':
            os.remove(os.path.join(root, file))

source_files = [];
for root, dirs, files in os.walk("..//..//src"):
    for file in files:
        if file.endswith(".cpp") and  file != '_pcl.cpp':
            source_files.append(os.path.join(root, file))
            print(os.path.join(root, file))

source_files.insert(0, '_pcl.pyx')


icl_dirs = ['..//..//src', '..//..//3rdParty//eigen', '..//..//3rdParty']
icl_dirs.append(numpy.get_include())
if os.name == 'nt':
    print('platform: nt')
    icl_dirs.append('..//..//3rdParty//msvc14//Qhull//include')
elif os.name == 'posix':
    icl_dirs.append('..//..//3rdParty//gcc//QHull//include')
else:
    print('not support platform!')
    exit(1)

lib_dirs = []
if os.name == 'nt':
    lib_dirs.append('..//..//3rdParty//msvc14//QHull//lib')
elif os.name == 'posix':
    lib_dirs.append('..//..//3rdParty//gcc//QHull//lib')
else:
    print('not support platform!')
    exit(1)

module = Extension('pcl._pcl',
                sources = source_files,
                include_dirs =  icl_dirs,
                library_dirs = lib_dirs,
                libraries = ['qhullstatic'],
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
