# -*- coding: utf-8 -*-
cimport pcl_surface_180 as pclsf
cimport pcl_defs as cpp
import math
cdef class ConcaveHull:
    """
    ConcaveHull class for ...
    """
    cdef pclsf.ConcaveHull_t *me

    def __cinit__(self):
        self.me = new pclsf.ConcaveHull_t()

    def __dealloc__(self):
        del self.me

    def set_input_cloud(self, PointCloud pc not None):
        (<cpp.PCLBase_t*>self.me).setInputCloud(pc.thisptr_shared)
    
    def reconstruct(self):
        """
        Apply the filter according to the previously set parameters and return
        a new pointcloud
        """
        cdef PointCloud pc = PointCloud()
        self.me.reconstruct(pc.thisptr()[0])
        return pc

    def reconstruct_contours(self):
        cdef PointCloud pc = PointCloud()
        cdef vector[cpp.Vertices] cpp_contours = vector[cpp.Vertices]()
        self.me.reconstruct(pc.thisptr()[0], cpp_contours)
        contours = []
        for cpp_contour in cpp_contours:
            len = cpp_contour.vertices.size()
            ct = np.zeros(len, dtype = np.int32)
            for i in range(len):
                ct[i] = cpp_contour.vertices[i]

            contours.append(ct)

        return pc, contours

    def reconstruct_contours_perimeters(self):
        cdef PointCloud pc = PointCloud()
        cdef vector[cpp.Vertices] cpp_contours = vector[cpp.Vertices]()

        self.me.reconstruct(pc.thisptr()[0], cpp_contours)

        contours = []
        perimeters = []
        for cpp_contour in cpp_contours:
            len = cpp_contour.vertices.size()
            ct = np.zeros(len, dtype = np.int32)
            perimeter = 0
            for i in range(len):
                v0 = cpp_contour.vertices[i]
                v1 = cpp_contour.vertices[ (i+1) % len]
                ct[i] = v0
                perimeter += self.distance(pc[v0],pc[v1])

            contours.append(ct)
            perimeters.append(perimeter)

        return pc, contours, perimeters

    def distance(self, p0, p1):
        p = (p1[0]-p0[0], p1[1]-p0[1], p1[2]-p0[2])
        return math.sqrt(p[0]*p[0] + p[1]*p[1] + p[2]*p[2])

    #def get_perimeter(self):
    #    return self.me.getPerimeter()

    def set_alpha(self, double dimension):
        self.me.setAlpha(dimension)

    def get_alpha(self):
        return self.me.getAlpha()

    def set_dimension (self, int dimension):
        self.me.setDimension(dimension)

    def get_dimension (self):
        return self.me.getDimension()

# cdef class ConcaveHull_PointXYZI:
#     """
#     ConcaveHull class for ...
#     """
#     cdef pclsf.ConcaveHull_PointXYZI_t *me
#     def __cinit__(self):
#         self.me = new pclsf.ConcaveHull_PointXYZI_t()
#     def __dealloc__(self):
#         del self.me
#
#     def reconstruct(self):
#         """
#         Apply the filter according to the previously set parameters and return
#         a new pointcloud
#         """
#         cdef PointCloud_PointXYZI pc = PointCloud_PointXYZI()
#         self.me.reconstruct(pc.thisptr()[0])
#         return pc
#
#     def set_Alpha(self, double d):
#         self.me.setAlpha (d)
#
#
# cdef class ConcaveHull_PointXYZRGB:
#     """
#     ConcaveHull class for ...
#     """
#     cdef pclsf.ConcaveHull_PointXYZRGB_t *me
#     def __cinit__(self):
#         self.me = new pclsf.ConcaveHull_PointXYZRGB_t()
#     def __dealloc__(self):
#         del self.me
#
#     def reconstruct(self):
#         """
#         Apply the filter according to the previously set parameters and return
#         a new pointcloud
#         """
#         cdef PointCloud_PointXYZRGB pc = PointCloud_PointXYZRGB()
#         self.me.reconstruct(pc.thisptr()[0])
#         return pc
#
#     def set_Alpha(self, double d):
#         self.me.setAlpha (d)
#
#
# cdef class ConcaveHull_PointXYZRGBA:
#     """
#     ConcaveHull class for ...
#     """
#     cdef pclsf.ConcaveHull_PointXYZRGBA_t *me
#     def __cinit__(self):
#         self.me = new pclsf.ConcaveHull_PointXYZRGBA_t()
#     def __dealloc__(self):
#         del self.me
#
#     def reconstruct(self):
#         """
#         Apply the filter according to the previously set parameters and return
#         a new pointcloud
#         """
#         cdef PointCloud_PointXYZRGBA pc = PointCloud_PointXYZRGBA()
#         self.me.reconstruct(pc.thisptr()[0])
#         return pc
#
#     def set_Alpha(self, double d):
#         self.me.setAlpha (d)
#
#