import numpy as np
import random
import pcl
from objsimple2 import objreader

reader = objreader.read('G:\\Projects\\Oh\\data\\test_data\\lucy_none-Slice-55_center_vn_normal.obj')

points = np.zeros((len(reader.vv), 3), dtype=np.float32)
normals = np.zeros((len(reader.vv),4), dtype=np.float32)

for v, vdata in enumerate(reader.vv):
    for i, scalar in enumerate(vdata):
        points[v][i] = scalar

for n, ndata in enumerate(reader._vn):
    for i, scalar in enumerate(ndata):
        normals[n][i] = scalar
    normals[n][3] = 1.0

cloudpoint = pcl.PointCloud()
cloudpoint.from_array(points)

cloudnormal = pcl.PointCloud_Normal()
cloudnormal.from_array(normals)

octreeNormal = pcl.OctreePointCloudNormal(0.1)
octreeNormal.set_input_cloud(cloudpoint)
octreeNormal.set_input_normal_cloud(cloudnormal)
octreeNormal.enable_dynamic_depth(100)
octreeNormal.set_leaf_normal_threshold(0.8)
octreeNormal.add_points_from_input_cloud()


RAND_MAX = 1024.0
searchPoint = pcl.PointCloud()
searchPoints = np.zeros((1, 3), dtype=np.float32)
searchPoints[0][0] = 1024 * random.random () / (RAND_MAX + 1.0)
searchPoints[0][1] = 1024 * random.random () / (RAND_MAX + 1.0)
searchPoints[0][2] = 1024 * random.random () / (RAND_MAX + 1.0)
searchPoint.from_array(searchPoints)


ind = octreeNormal.VoxelSearch(searchPoint)

print ('Neighbors within voxel search at (' + str(searchPoint[0][0]) + ' ' + str(searchPoint[0][1]) + ' ' + str(searchPoint[0][2]) + ')')
for i in range(0, ind.size):
    print ('index = ' + str(ind[i]))
    print ('(' + str(cloudpoint[ind[i]][0]) + ' ' + str(cloudpoint[ind[i]][1]) + ' ' + str(cloudpoint[ind[i]][2]))
