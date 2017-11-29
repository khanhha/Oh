# -*- coding: utf-8 -*-
cimport pcl_defs as cpp
cimport pcl_octree_180 as pcloct

cdef class OctreePointCloudNormal(OctreePointCloud):
    """
    Octree pointcloud normal
    """
    cdef pcloct.OctreePointCloudNormal_t *me2

    def __cinit__(self, double resolution):
        """
        Constructs octree pointcloud with given resolution at lowest octree level
        """
        self.me2 = NULL
        self.me = NULL
        if resolution <= 0.:
            raise ValueError("Expected resolution > 0., got %r" % resolution)

        self.me2 = <pcloct.OctreePointCloudNormal_t*> new pcloct.OctreePointCloudNormal_t(resolution)
        self.me = <pcloct.OctreePointCloud_t*> self.me2

    def __dealloc__(self):
        del self.me2
        self.me2 = NULL
        self.me = NULL

    def enable_dynamic_depth(self, int max_obj_per_leaf):
         (<pcloct.OctreePointCloud_t*>self.me).enableDynamicDepth(max_obj_per_leaf)

    def set_input_normal_cloud(self, PointCloud_Normal pc):
        """
        Provide a pointer to the input data set.
        """
        self.me2.setInputNormalCloud(pc.thisptr_shared)

    def set_leaf_normal_threshold(self, double value):
        self.me2.setNormalThreshold(value)

    # base OctreePointCloud
    def define_bounding_box(self):
        """
        Investigate dimensions of pointcloud data set and define corresponding bounding box for octree.
        """
        self.me.defineBoundingBox()

    # def define_bounding_box(self, double min_x, double min_y, double min_z, double max_x, double max_y, double max_z):
    #     """
    #     Define bounding box for octree. Bounding box cannot be changed once the octree contains elements.
    #     """
    #     self.me2.defineBoundingBox(min_x, min_y, min_z, max_x, max_y, max_z)

    def add_points_from_input_cloud(self):
        """
        Add points from input point cloud to octree.
        """
        self.me.addPointsFromInputCloud()



