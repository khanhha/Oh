# -*- coding: utf-8 -*-
cimport pcl_defs as cpp
cimport pcl_kdtree_180 as pclkdt

cdef class KdTree:
    """
    Finds k nearest neighbours from points in another pointcloud to points in
    a reference pointcloud.

    Must be constructed from the reference point cloud, which is copied, so
    changed to pc are not reflected in KdTree(pc).
    """
    cdef pclkdt.KdTree_t *me

    def __cinit__(self, PointCloud pc not None):
        self.me = new pclkdt.KdTree_t()
        self.me.setInputCloud(pc.thisptr_shared)

    def __dealloc__(self):
        del self.me


