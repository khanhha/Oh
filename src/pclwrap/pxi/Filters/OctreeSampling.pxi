cimport pcl_defs as cpp
cimport pcl_filters_180 as pclfil
from enum import IntEnum

class ResampleMethod(IntEnum):
    UNIFORM = 0
    NONUNIFORM_MAX_POINTS_PER_LEAF = 1
    NONUNIFORM_NORMAL_THRESHOLD = 2

class InterpolationMethod(IntEnum):
    CLOSEST_TO_CENTER = 0
    AVERAGE = 1
    HEIGHT_INTERPOLATION = 2

cdef class OctreeSampling:
    """
    Assembles a local 3D grid over a given PointCloud, and downsamples + filters the data.
    """
    cdef pclfil.OctreeSampling_t *me
    def __cinit__(self):
        self.me = new pclfil.OctreeSampling_t()
    def __dealloc__(self):
        del self.me

    def set_input_cloud(self, PointCloud pc not None):
        (<cpp.PCLBase_t*>self.me).setInputCloud(pc.thisptr_shared)

    def set_input_normal_cloud(self, PointCloud_Normal pc not None):
        self.me.setInputNormalCloud(pc.thisptr_shared)

    def filter(self):
        """
        Apply the filter according to the previously set parameters and return
        a new pointcloud
        """
        cdef PointCloud pc = PointCloud()
        self.me.filter(pc.thisptr()[0])
        return pc

    def set_octree_resolution(self, double rel):
        self.me.setOctreeResolution(rel)

    def set_octree_normal_threshold(self, double thres):
        self.me.setOctreeNormalThreshold(thres)

    def set_sampling_resolution(self, double rel):
        self.me.setSamplingResolution(rel)

    def set_max_points_per_leaf(self, size_t n):
        self.me.setMaxPointsPerLeaf(n)

    def set_sample_radius_search(self, double radius):
        self.me.setSampleRadiusSearch(radius)

    def set_resample_method(self, method):
        self.me.setResampleMethod(method)

    def set_interpolation_method(self, method):
        self.me.setInterpolationMethod(method)