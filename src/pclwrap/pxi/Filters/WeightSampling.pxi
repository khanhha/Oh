cimport pcl_defs as cpp
cimport pcl_filters_180 as pclfil

cdef class WeightSampling:
    """
    Assembles a local 3D grid over a given PointCloud, and downsamples + filters the data.
    """
    cdef pclfil.WeightSampling_t *me

    def __cinit__(self):
        self.me = new pclfil.WeightSampling_t()

    def __dealloc__(self):
        del self.me

    def set_input_cloud(self, PointCloud pc not None):
        (<cpp.PCLBase_t*>self.me).setInputCloud(pc.thisptr_shared)

    def filter(self):
        """
        Apply the filter according to the previously set parameters and return
        a new pointcloud
        """
        cdef PointCloud pc = PointCloud()
        self.me.filter(pc.thisptr()[0])
        return pc

    def set_resample_percent(self,float percent):
        self.me.setResamplePercent(percent)

    def set_KNeighbour_search(self,int k):
        self.me.setKNeighbourSearch(k)

    def set_radius_search(self,float radius):
        self.me.setRadiusSearch(radius)

    def set_sigma(self,float sig):
        self.me.setSigma(sig)