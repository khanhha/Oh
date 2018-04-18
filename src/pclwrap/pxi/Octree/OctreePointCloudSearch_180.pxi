# -*- coding: utf-8 -*-
cimport pcl_defs as cpp
cimport pcl_octree_180 as pcloct

# include "PointXYZtoPointXYZ.pxi" --> multiple define ng
# include "OctreePointCloud.pxi"

cdef class OctreePointCloudSearch(OctreePointCloud):
    """
    Octree pointcloud search
    """
    cdef pcloct.OctreePointCloudSearch_t *me2

    def __cinit__(self, double resolution):
        """
        Constructs octree pointcloud with given resolution at lowest octree level
        """ 
        self.me2 = NULL
        self.me = NULL
        if resolution <= 0.:
            raise ValueError("Expected resolution > 0., got %r" % resolution)

        self.me2 = <pcloct.OctreePointCloudSearch_t*> new pcloct.OctreePointCloudSearch_t(resolution)
        self.me = <pcloct.OctreePointCloud_t*> self.me2

    def __dealloc__(self):
        del self.me2
        self.me2 = NULL
        self.me = NULL

    def enable_dynamic_depth(self, int max_obj_per_leaf):
         (<pcloct.OctreePointCloud_t*>self.me).enableDynamicDepth(max_obj_per_leaf)

    # nearestKSearch
    ###
    def nearest_k_search_for_cloud(self, PointCloud pc not None, int k=1):
        """
        Find the k nearest neighbours and squared distances for all points
        in the pointcloud. Results are in ndarrays, size (pc.size, k)
        Returns: (k_indices, k_sqr_distances)
        """
        cdef cnp.npy_intp n_points = pc.size
        cdef cnp.ndarray[float, ndim=2] sqdist = np.zeros((n_points, k),
                                                          dtype=np.float32)
        cdef cnp.ndarray[int, ndim=2] ind = np.zeros((n_points, k),
                                                     dtype=np.int32)

        for i in range(n_points):
            self._nearest_k(pc, i, k, ind[i], sqdist[i])
        return ind, sqdist

    def nearest_k_search_for_point(self, PointCloud pc not None, int index, int k=1):
        """
        Find the k nearest neighbours and squared distances for the point
        at pc[index]. Results are in ndarrays, size (k)
        Returns: (k_indices, k_sqr_distances)
        """
        cdef cnp.ndarray[float] sqdist = np.zeros(k, dtype=np.float32)
        cdef cnp.ndarray[int] ind = np.zeros(k, dtype=np.int32)

        self._nearest_k(pc, index, k, ind, sqdist)
        return ind, sqdist

    def approx_nearest_search(self, point):
        cdef int ret_idx = -1
        cdef float ret_dst = 0
        (<pcloct.OctreePointCloudSearch_t*>self.me).approxNearestSearch(to_point_t(point), ret_idx, ret_dst)
        return ret_idx, ret_dst

    def box_search(self, b_min, b_max):
        cdef vector[int] k_indices
        cdef int k = (<pcloct.OctreePointCloudSearch_t*>self.me).boxSearch(to_point_t(b_min), to_point_t(b_max), k_indices)
        cdef cnp.ndarray[int] np_k_indices = np.zeros(k, dtype=np.int32)
        for i in range(k):
            np_k_indices[i] = k_indices[i]
        return np_k_indices

    @cython.boundscheck(False)
    cdef void _nearest_k(self, PointCloud pc, int index, int k,
                         cnp.ndarray[ndim=1, dtype=int, mode='c'] ind,
                         cnp.ndarray[ndim=1, dtype=float, mode='c'] sqdist
                        ) except +:
        # k nearest neighbors query for a single point.
        cdef vector[int] k_indices
        cdef vector[float] k_sqr_distances
        k_indices.resize(k)
        k_sqr_distances.resize(k)
        # self.me.nearestKSearch(pc.thisptr()[0], index, k, k_indices, k_sqr_distances)
        (<pcloct.OctreePointCloudSearch_t*>self.me).nearestKSearch(pc.thisptr()[0], index, k, k_indices, k_sqr_distances)

        for i in range(k):
            sqdist[i] = k_sqr_distances[i]
            ind[i] = k_indices[i]

    def nearest_k_search_for_a_point(self, point, int k=1):
        """
        Find the k nearest neighbours and squared distances for the point
        at pc[index]. Results are in ndarrays, size (k)
        Returns: (k_indices, k_sqr_distances)
        """
        cdef cnp.ndarray[float] sqdist = np.zeros(k, dtype=np.float32)
        cdef cnp.ndarray[int] ind = np.zeros(k, dtype=np.int32)

        self._nearest_k_1(point, k, ind, sqdist)
        return ind, sqdist

    @cython.boundscheck(False)
    cdef void _nearest_k_1(self, point, int k,
                         cnp.ndarray[ndim=1, dtype=int, mode='c'] ind,
                         cnp.ndarray[ndim=1, dtype=float, mode='c'] sqdist
                        ) except +:
        # k nearest neighbors query for a single point.
        cdef vector[int] k_indices
        cdef vector[float] k_sqr_distances
        k_indices.resize(k)
        k_sqr_distances.resize(k)
        # self.me.nearestKSearch(pc.thisptr()[0], index, k, k_indices, k_sqr_distances)
        (<pcloct.OctreePointCloudSearch_t*>self.me).nearestKSearch(to_point_t(point), k, k_indices, k_sqr_distances)

        for i in range(k):
            sqdist[i] = k_sqr_distances[i]
            ind[i] = k_indices[i]
    ###

    # radius Search
    ###
    def radius_search (self, point, double radius, unsigned int max_nn = 0):
        """
        Search for all neighbors of query point that are within a given radius.
        
        Returns: (k_indices, k_sqr_distances)
        """
        cdef vector[int] k_indices
        cdef vector[float] k_sqr_distances
        if max_nn > 0:
            k_indices.resize(max_nn)
            k_sqr_distances.resize(max_nn)
        cdef int k = (<pcloct.OctreePointCloudSearch_t*>self.me).radiusSearch(to_point_t(point), radius, k_indices, k_sqr_distances, max_nn)
        cdef cnp.ndarray[float] np_k_sqr_distances = np.zeros(k, dtype=np.float32)
        cdef cnp.ndarray[int] np_k_indices = np.zeros(k, dtype=np.int32)
        for i in range(k):
            np_k_sqr_distances[i] = k_sqr_distances[i]
            np_k_indices[i] = k_indices[i]
        return np_k_indices, np_k_sqr_distances

    ###

    # Voxel Search
    ### 
    def voxel_search(self, point):
        """
        Search for all neighbors of query point that are within a given voxel.
        
        Returns: (v_indices)
        """
        cdef vector[int] v_indices

        # print('before v_indices count = ' + str(v_indices.size()))
        self._VoxelSearch(to_point_t(point), v_indices)
        v = v_indices.size()
        # print('after v_indices count = ' + str(v))
        
        cdef cnp.ndarray[int] np_v_indices = np.zeros(v, dtype=np.int32)
        for i in range(v):
            np_v_indices[i] = v_indices[i]
        
        return np_v_indices

    @cython.boundscheck(False)
    cdef void _VoxelSearch(self, cpp.PointXYZ point, vector[int] &v_indices) except +:
        cdef vector[int] voxel_indices
        # k = 10
        # voxel_indices.resize(k)
        (<pcloct.OctreePointCloudSearch_t*>self.me).voxelSearch(point, voxel_indices)
        
        # print('_VoxelSearch k = ' + str(k))
        # print('_VoxelSearch voxel_indices = ' + str(voxel_indices.size()))
        k = voxel_indices.size()
        
        for i in range(k):
            v_indices.push_back(voxel_indices[i])

    ### 


#     def radius_search_for_cloud(self, PointCloud pc not None, double radius):
#         """
#         Find the radius and squared distances for all points
#         in the pointcloud. Results are in ndarrays, size (pc.size, k)
#         Returns: (radius_indices, radius_distances)
#         """
#         k = 10
#         cdef cnp.npy_intp n_points = pc.size
#         cdef cnp.ndarray[float, ndim=2] sqdist = np.zeros((n_points, k),
#                                                           dtype=np.float32)
#         cdef cnp.ndarray[int, ndim=2] ind = np.zeros((n_points, k),
#                                                      dtype=np.int32)
# 
#         for i in range(n_points):
#             self._search_radius(pc, i, k, radius, ind[i], sqdist[i])
#         return ind, sqdist
# 
#     @cython.boundscheck(False)
#     cdef void _search_radius(self, PointCloud pc, int index, int k, double radius,
#                          cnp.ndarray[ndim=1, dtype=int, mode='c'] ind,
#                          cnp.ndarray[ndim=1, dtype=float, mode='c'] sqdist
#                         ) except +:
#         # radius query for a single point.
#         cdef vector[int] radius_indices
#         cdef vector[float] radius_distances
#         radius_indices.resize(k)
#         radius_distances.resize(k)
#         # self.me.radiusSearch(pc.thisptr()[0], index, radius, radius_indices, radius_distances)
#         k = (<pcloct.OctreePointCloudSearch_t*>self.me).radiusSearch(pc.thisptr()[0], index, radius, radius_indices, radius_distances, 10)
# 
#         for i in range(k):
#             sqdist[i] = radius_distances[i]
#             ind[i] = radius_indices[i]

    # base OctreePointCloud
    def define_bounding_box(self):
        """
        Investigate dimensions of pointcloud data set and define corresponding bounding box for octree. 
        """
        self.me2.defineBoundingBox()

    # def define_bounding_box(self, double min_x, double min_y, double min_z, double max_x, double max_y, double max_z):
    #     """
    #     Define bounding box for octree. Bounding box cannot be changed once the octree contains elements.
    #     """
    #     self.me2.defineBoundingBox(min_x, min_y, min_z, max_x, max_y, max_z)

    def add_points_from_input_cloud(self):
        """
        Add points from input point cloud to octree.
        """
        self.me2.addPointsFromInputCloud()

    def is_voxel_occupied_at_point(self, point):
        """
        Check if voxel at given point coordinates exist.
        """
        return self.me2.isVoxelOccupiedAtPoint(point[0], point[1], point[2])

    def get_occupied_voxel_centers(self):
        """
        Get list of centers of all occupied voxels.
        """
        cdef eig.AlignedPointTVector_t points_v
        cdef int num = self.me2.getOccupiedVoxelCenters (points_v)
        return [(points_v[i].x, points_v[i].y, points_v[i].z) for i in range(num)]

    def delete_voxel_at_point(self, point):
        """
        Delete leaf node / voxel at given point.
        """
        self.me2.deleteVoxelAtPoint(to_point_t(point))
