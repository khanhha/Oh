
cdef class SlicePerimeter:
    cdef pclsf.ConvexHull_t *me

    def __cinit__(self):
        self.me = new pclsf.ConvexHull_t()

    def __dealloc__(self):
        del self.me

    def measure(self, PointCloud pc not None):
        