Install steps
1. Install Python 3.5. X64 version. Note, the code currently does not support Python 32 bit
2. Install latest cython version. tested on version 0.26.1 and 0.27.3. There're still bugs on the latest version: 0.28.1
3. Install numpy
4. start the command line window.
5. [linux] install build tools: sudo apt-get install build-essential (make sure that you have the latest g++ version: 5.4)
6. move the folder /src/pclwrap
7. run the command: sudo python setup.py install
8. run the example src/pclwrap/octree_test.py


Install VTK for octree visualization (In case you want to run octree visualization)
 - download VTK 8.0.1  from
    - windows: https://www.lfd.uci.edu/~gohlke/pythonlibs/
    - linux: https://pypi.python.org/pypi/vtk
 - guide to install vtk using whl file:
    - run command: pip install vtk-8.0.0.dev20170717-cp36-cp36m-manylinux1_x86_64.whl
    - for linux: run the following command: export LD_LIBRARY_PATH=/home/khanh/anaconda3/lib/python3.6/site-packages/vtk

