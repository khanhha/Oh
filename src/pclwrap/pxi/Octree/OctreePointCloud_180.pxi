# -*- coding: utf-8 -*-
cimport pcl_defs as cpp
cimport pcl_octree_180 as pcloct
import numpy as np
cimport numpy as cnp

cimport eigen as eig

# cdef class OctreeKey:
#     cdef unsigned int x
#     cdef unsigned int y
#     cdef unsigned int z
#     def __init__(self, unsigned int k0, unsigned int k1, unsigned int k2):
#         self.x = k0
#         self.y = k1
#         self.z = k2

cdef class OctreePointCloud:
    """
    Octree pointcloud
    """
    cdef pcloct.OctreePointCloud_t *me

    # def __cinit__(self, double resolution):
    #     self.me = NULL
    #     if resolution <= 0.:
    #         raise ValueError("Expected resolution > 0., got %r" % resolution)

    # NG(BUild Error)
    # def __init__(self, double resolution):
    #     """
    #     Constructs octree pointcloud with given resolution at lowest octree level
    #     """ 
    #     cdef double param = 0
    #     self.me = new pcloct.OctreePointCloud_t(0)
    #     # self.me = new pcloct.OctreePointCloud_t(resolution)
    #     # self.me = new pcloct.OctreePointCloud_t()

    # def __dealloc__(self):
    #     del self.me
    #     self.me = NULL      # just to be sure

    def __cinit__(self, double resolution):
        """
        Constructs octree pointcloud with given resolution at lowest octree level
        """
        self.me = NULL
        if resolution <= 0.:
            raise ValueError("Expected resolution > 0., got %r" % resolution)

        self.me = <pcloct.OctreePointCloud_t*> new pcloct.OctreePointCloud_t(resolution)

    def __dealloc__(self):
        del self.me
        self.me = NULL

    def set_input_cloud(self, PointCloud pc):
        """
        Provide a pointer to the input data set.
        """
        self.me.setInputCloud(pc.thisptr_shared)

    # def define_bounding_box(self):
    #     """
    #     Investigate dimensions of pointcloud data set and define corresponding bounding box for octree. 
    #     """
    #     self.me.defineBoundingBox()
        
    # def define_bounding_box(self, double min_x, double min_y, double min_z, double max_x, double max_y, double max_z):
    #     """
    #     Define bounding box for octree. Bounding box cannot be changed once the octree contains elements.
    #     """
    #     self.me.defineBoundingBox(min_x, min_y, min_z, max_x, max_y, max_z)

    def add_points_from_input_cloud(self):
    #     """
    #     Add points from input point cloud to octree.
    #     """
         self.me.addPointsFromInputCloud()

    def get_bounding_box(self):
        cdef double minx = -1, miny = -1, minz= -1, maxx = -1, maxy = -1, maxz = -1
        self.me.getBoundingBox(minx, miny, minz, maxx, maxy, maxz)
        return [minx, miny, minz],[maxx, maxy, maxz]

    def delete_tree(self):
        """
        Delete the octree structure and its leaf nodes.
        """
        self.me.deleteTree()

    # def is_voxel_occupied_at_point(self, point):
    #     """
    #     Check if voxel at given point coordinates exist.
    #     """
    #     return self.me.isVoxelOccupiedAtPoint(point[0], point[1], point[2])

    # def get_occupied_voxel_centers(self):
    #     """
    #     Get list of centers of all occupied voxels.
    #     """
    #     cdef eig.AlignedPointTVector_t points_v
    #     cdef int num = self.me.getOccupiedVoxelCenters (points_v)
    #     # cdef int num = self.me.getOccupiedVoxelCenters (<eig.AlignedPointTVector_t> points_v)
    #     # cdef int num = mpcl_getOccupiedVoxelCenters(self.me, points_v)
    #     # cdef int num = mpcl_getOccupiedVoxelCenters(deref(self.me), points_v)
    #     return [(points_v[i].x, points_v[i].y, points_v[i].z) for i in range(num)]

    # def delete_voxel_at_point(self, point):
    #     """
    #     Delete leaf node / voxel at given point.
    #     """
    #     self.me.deleteVoxelAtPoint(to_point_t(point))
    #     # mpcl_deleteVoxelAtPoint(self.me, to_point_t(point))
    #     # mpcl_deleteVoxelAtPoint(deref(self.me), to_point_t(point))

    def get_leaf_count(self):
        cdef size_t count = (<pcloct.OctreeBase_OctreeContainerPointIndices_t*>self.me).getLeafCount()
        return count

    def get_branch_count(self):
        cdef size_t count = (<pcloct.OctreeBase_OctreeContainerPointIndices_t*>self.me).getBranchCount()
        return count

    def get_tree_depth(self):
        cdef size_t depth = (<pcloct.OctreeBase_OctreeContainerPointIndices_t*>self.me).getTreeDepth()
        return depth

    def gell_all_node_keys_at_max_depth(self, depth_arg):
        cdef vector[pcloct.OctreeKey] keys
        cdef vector[int] depths

        self.me.getOctreeKeysAtMaxDepth(depth_arg, keys, depths)

        cdef int len = keys.size()

        ret_keys = np.zeros((len, 3), dtype=np.int32)
        ret_depths = np.zeros(len, dtype=np.int32)

        for i in range(len):
            ret_keys[i][0] = keys[i].x
            ret_keys[i][1] = keys[i].y
            ret_keys[i][2] = keys[i].z

            ret_depths[i] = depths[i]

        return ret_keys, ret_depths

    def gell_all_node_keys_at_depth(self, depth_arg):
        cdef vector[pcloct.OctreeKey] keys
        cdef vector[int] depths

        self.me.getOctreeKeysAtDepth(depth_arg, keys, depths)

        cdef int len = keys.size()

        ret_keys = np.zeros((len, 3), dtype=np.int32)
        ret_depths = np.zeros(len, dtype=np.int32)

        for i in range(len):
            ret_keys[i][0] = keys[i].x
            ret_keys[i][1] = keys[i].y
            ret_keys[i][2] = keys[i].z

            ret_depths[i] = depths[i]

        return ret_keys, ret_depths

    def get_all_leaf_keys(self):

        cdef vector[pcloct.OctreeKey] keys
        cdef vector[int] depths

        self.me.getAllLeafKeys(keys, depths)

        cdef int len = keys.size()

        ret_keys = np.zeros((len, 3), dtype=np.int32)
        ret_depths = np.zeros(len, dtype=np.int32)

        for i in range(len):
            ret_keys[i][0] = keys[i].x
            ret_keys[i][1] = keys[i].y
            ret_keys[i][2] = keys[i].z

            ret_depths[i] = depths[i]

        return ret_keys, ret_depths

    def gen_voxel_center_from_octree_key(self, key, unsigned int depth):
        cdef pcloct.OctreeKey key_ = pcloct.OctreeKey(np.uint(key[0]), np.uint(key[1]), np.uint(key[2]))
        cdef cpp.PointXYZ p

        self.me.genVoxelCenterFromOctreeKey(key_, depth, p)

        return [p.x, p.y, p.z]

    def gen_voxel_bounds_from_octree_key(self, key, depth):
        cdef pcloct.OctreeKey key_ = pcloct.OctreeKey(np.int(key[0]), np.int(key[1]), np.int(key[2]))
        cdef eig.Vector3f bmin
        cdef eig.Vector3f bmax

        self.me.genVoxelBoundsFromOctreeKey(key_, depth, bmin, bmax)

        return [bmin.data()[0], bmin.data()[1], bmin.data()[2]], [bmax.data()[0], bmax.data()[1], bmax.data()[2]]

cdef class OctreePointCloud_PointXYZI:
    """
    Octree pointcloud
    """
    cdef pcloct.OctreePointCloud_PointXYZI_t *me

    # def __cinit__(self, double resolution):
    #     self.me = NULL
    #     if resolution <= 0.:
    #         raise ValueError("Expected resolution > 0., got %r" % resolution)

    # NG(BUild Error)
    # def __init__(self, double resolution):
    #     """
    #     Constructs octree pointcloud with given resolution at lowest octree level
    #     """ 
    #     cdef double param = 0
    #     # self.me = new pcloct.OctreePointCloud_PointXYZI_t(param)
    #     # self.me = new pcloct.OctreePointCloud_PointXYZI_t(resolution)
    #     # self.me = new pcloct.OctreePointCloud_PointXYZI_t()

    # def __dealloc__(self):
    #     del self.me
    #     self.me = NULL      # just to be sure

    def set_input_cloud(self, PointCloud_PointXYZI pc):
        """
        Provide a pointer to the input data set.
        """
        self.me.setInputCloud(pc.thisptr_shared)

    # def define_bounding_box(self):
    #     """
    #     Investigate dimensions of pointcloud data set and define corresponding bounding box for octree. 
    #     """
    #     self.me.defineBoundingBox()

    # def define_bounding_box(self, double min_x, double min_y, double min_z, double max_x, double max_y, double max_z):
    #     """
    #     Define bounding box for octree. Bounding box cannot be changed once the octree contains elements.
    #     """
    #     self.me.defineBoundingBox(min_x, min_y, min_z, max_x, max_y, max_z)

    # def add_points_from_input_cloud(self):
    #     """
    #     Add points from input point cloud to octree.
    #     """
    #     self.me.addPointsFromInputCloud()

    def delete_tree(self):
        """
        Delete the octree structure and its leaf nodes.
        """
        self.me.deleteTree()

    # def is_voxel_occupied_at_point(self, point):
    #     """
    #     Check if voxel at given point coordinates exist.
    #     """
    #     return self.me.isVoxelOccupiedAtPoint(point[0], point[1], point[2])

    # def get_occupied_voxel_centers(self):
    #     """
    #     Get list of centers of all occupied voxels.
    #     """
    #     cdef eig.AlignedPointTVector_PointXYZI_t points_v
    #     cdef int num = self.me.getOccupiedVoxelCenters (points_v)
    #     # cdef int num = self.me.getOccupiedVoxelCenters (<eig.AlignedPointTVector_PointXYZI_t> points_v)
    #     # cdef int num = mpcl_getOccupiedVoxelCenters(self.me, points_v)
    #     # cdef int num = mpcl_getOccupiedVoxelCenters_PointXYZI(deref(self.me), points_v)
    #     return [(points_v[i].x, points_v[i].y, points_v[i].z) for i in range(num)]

    # def delete_voxel_at_point(self, point):
    #     """
    #     Delete leaf node / voxel at given point.
    #     """
    #     # NG (use minipcl?)
    #     self.me.deleteVoxelAtPoint(to_point2_t(point))
    #     # mpcl_deleteVoxelAtPoint(self.me, to_point2_t(point))
    #     # mpcl_deleteVoxelAtPoint_PointXYZI(deref(self.me), to_point2_t(point))


cdef class OctreePointCloud_PointXYZRGB:
    """
    Octree pointcloud
    """
    cdef pcloct.OctreePointCloud_PointXYZRGB_t *me

    # def __cinit__(self, double resolution):
    #     self.me = NULL
    #     if resolution <= 0.:
    #         raise ValueError("Expected resolution > 0., got %r" % resolution)

    # NG(BUild Error)
    # def __init__(self, double resolution):
    #     """
    #     Constructs octree pointcloud with given resolution at lowest octree level
    #     """ 
    #     cdef double param = 0
    #     self.me = new pcloct.OctreePointCloud_PointXYZRGB_t(param)
    #     # self.me = new pcloct.OctreePointCloud_PointXYZRGB_t(resolution)
    #     # self.me = new pcloct.OctreePointCloud_PointXYZRGB_t()

    # def __dealloc__(self):
    #     del self.me
    #     self.me = NULL      # just to be sure

    def set_input_cloud(self, PointCloud_PointXYZRGB pc not None):
        """
        Provide a pointer to the input data set.
        """
        self.me.setInputCloud(pc.thisptr_shared)

    # def define_bounding_box(self):
    #     """
    #     Investigate dimensions of pointcloud data set and define corresponding bounding box for octree. 
    #     """
    #     self.me.defineBoundingBox()

    # def define_bounding_box(self, double min_x, double min_y, double min_z, double max_x, double max_y, double max_z):
    #     """
    #     Define bounding box for octree. Bounding box cannot be changed once the octree contains elements.
    #     """
    #     self.me.defineBoundingBox(min_x, min_y, min_z, max_x, max_y, max_z)

    # def add_points_from_input_cloud(self):
    #     """
    #     Add points from input point cloud to octree.
    #     """
    #     self.me.addPointsFromInputCloud()

    def delete_tree(self):
        """
        Delete the octree structure and its leaf nodes.
        """
        self.me.deleteTree()

    # def is_voxel_occupied_at_point(self, point):
    #     """
    #     Check if voxel at given point coordinates exist.
    #     """
    #     return self.me.isVoxelOccupiedAtPoint(point[0], point[1], point[2])

    # def get_occupied_voxel_centers(self):
    #     """
    #     Get list of centers of all occupied voxels.
    #     """
    #     cdef eig.AlignedPointTVector_PointXYZRGB_t points_v
    #     cdef int num = self.me.getOccupiedVoxelCenters (points_v)
    #     # cdef int num = mpcl_getOccupiedVoxelCenters(self.me, points_v)
    #     return [(points_v[i].x, points_v[i].y, points_v[i].z) for i in range(num)]

    # def delete_voxel_at_point(self, point):
    #     """
    #     Delete leaf node / voxel at given point.
    #     """
    #     # NG (minipcl?)
    #     self.me.deleteVoxelAtPoint(to_point3_t(point))


cdef class OctreePointCloud_PointXYZRGBA:
    """
    Octree pointcloud
    """
    cdef pcloct.OctreePointCloud_PointXYZRGBA_t *me

    # def __cinit__(self, double resolution):
    #     self.me = NULL
    #     if resolution <= 0.:
    #         raise ValueError("Expected resolution > 0., got %r" % resolution)

    # NG(BUild Error)
    # def __init__(self, double resolution):
    #     """
    #     Constructs octree pointcloud with given resolution at lowest octree level
    #     """ 
    #     cdef double param = 0
    #     self.me = new pcloct.OctreePointCloud_PointXYZRGBA_t(param)
    #     # self.me = new pcloct.OctreePointCloud_PointXYZRGBA_t(resolution)
    #     # self.me = new pcloct.OctreePointCloud_PointXYZRGBA_t()

    # def __dealloc__(self):
    #     del self.me
    #     self.me = NULL      # just to be sure

    def set_input_cloud(self, PointCloud_PointXYZRGBA pc):
        """
        Provide a pointer to the input data set.
        """
        self.me.setInputCloud(pc.thisptr_shared)

    # def define_bounding_box(self):
    #     """
    #     Investigate dimensions of pointcloud data set and define corresponding bounding box for octree. 
    #     """
    #     self.me.defineBoundingBox()

    # def define_bounding_box(self, double min_x, double min_y, double min_z, double max_x, double max_y, double max_z):
    #     """
    #     Define bounding box for octree. Bounding box cannot be changed once the octree contains elements.
    #     """
    #     self.me.defineBoundingBox(min_x, min_y, min_z, max_x, max_y, max_z)

    # use NG
    # def add_points_from_input_cloud(self):
    #     """
    #     Add points from input point cloud to octree.
    #     """
    #     self.me.addPointsFromInputCloud()

    def delete_tree(self):
        """
        Delete the octree structure and its leaf nodes.
        """
        self.me.deleteTree()

    # def is_voxel_occupied_at_point(self, point):
    #     """
    #     Check if voxel at given point coordinates exist.
    #     """
    #     return self.me.isVoxelOccupiedAtPoint(point[0], point[1], point[2])

    # def get_occupied_voxel_centers(self):
    #     """
    #     Get list of centers of all occupied voxels.
    #     """
    #     cdef eig.AlignedPointTVector_PointXYZRGBA_t points_v
    #     cdef int num = self.me.getOccupiedVoxelCenters (points_v)
    #     # cdef int num = mpcl_getOccupiedVoxelCenters(self.me, points_v)
    #     return [(points_v[i].x, points_v[i].y, points_v[i].z) for i in range(num)]

    # def delete_voxel_at_point(self, point):
    #     """
    #     Delete leaf node / voxel at given point.
    #     """
    #     # NG (minipcl?)
    #     self.me.deleteVoxelAtPoint(to_point4_t(point))


