# -*- coding: utf-8 -*-
cimport pcl_surface_180 as pclsf
cimport pcl_defs as cpp

cdef class ConvexHull:
    """
    ConcaveHull class for ...
    """
    cdef pclsf.ConvexHull_t *me

    def __cinit__(self):
        self.me = new pclsf.ConvexHull_t()

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

    def set_compute_area_volume (self, bool value):
        self.me.setComputeAreaVolume(value)

    def get_total_area (self):
        return self.me.getTotalArea()

    def get_perimeter(self):
        return self.me.getPerimeter()

    def get_projection_dimension(self):
        return self.me.getProjectionDimension()

    def get_total_volume (self):
        return self.me.getTotalVolume()

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