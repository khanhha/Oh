import numpy as np
import random
import pcl
from objsimple2 import objreader

# load point and normals
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

# create point cloud cloud
cloudpoint = pcl.PointCloud()
cloudpoint.from_array(points)
cloudnormal = pcl.PointCloud_Normal()
cloudnormal.from_array(normals)

# build tree based on normal and max per leaf node
octreeNormal = pcl.OctreePointCloudNormal(0.1)
octreeNormal.set_input_cloud(cloudpoint)
octreeNormal.set_input_normal_cloud(cloudnormal)
octreeNormal.enable_dynamic_depth(100)
octreeNormal.set_leaf_normal_threshold(0.8)
octreeNormal.add_points_from_input_cloud()

# test search method
searchPoint = cloudpoint[50]
[ind, sqdist] = octreeNormal.radius_search(searchPoint, 30, 20)

# nearest k search
[ind_1, sqdist_1] = octreeNormal.nearest_k_search_for_a_point(searchPoint, 20)

# nearest point
[idx_2, dst_2] = octreeNormal.approx_nearest_search(cloudpoint[50]);
assert idx_2 == 50

# bounding box search
[bmin, bmax] = octreeNormal.get_bounding_box()
idx_3 = octreeNormal.box_search(bmin, bmax)

# voxel search
ind = octreeNormal.voxel_search(searchPoint)

print ('Neighbors within voxel search at (' + str(searchPoint[0][0]) + ' ' + str(searchPoint[0][1]) + ' ' + str(searchPoint[0][2]) + ')')
for i in range(0, ind.size):
    print ('index = ' + str(ind[i]))
    print ('(' + str(cloudpoint[ind[i]][0]) + ' ' + str(cloudpoint[ind[i]][1]) + ' ' + str(cloudpoint[ind[i]][2]))
