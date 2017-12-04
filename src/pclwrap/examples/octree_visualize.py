import os.path
import numpy as np
import pcl
from objsimple2 import objreader

basepath = 'G:\\Projects\\Oh\data\\test_data\\'
filename = 'normal_lucy_none-Slice-54_center_vn.obj'

filepath = basepath + filename

if os.path.exists(filepath) == False:
    print('Error file does not exist: ' + filepath)

reader = objreader.read(filepath)
points = np.zeros((len(reader.vv), 3), dtype=np.float32)
normals = np.zeros((len(reader.vv), 4), dtype=np.float32)
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

[keys, depths] = octreeNormal.get_all_leaf_keys()
for k in range(len(keys)):
    print('{} {} {} {}'.format(depths[k], keys[k][0], keys[k][1], keys[k][2]))
