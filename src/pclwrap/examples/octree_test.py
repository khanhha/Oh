import os.path
import numpy as np
import pcl

from objsimple2 import objreader

basepath = 'D:\\Projects\\Oh\data\\test_data\\'
filenames = {'normal_lucy_none-Slice-54_center_vn.obj',
             'normal_lucy_none-Slice-55_center_vn.obj',
             'normal_lucy_none-Slice-56_center_vn.obj',
             'normal_lucy_none-Slice-57_center_vn.obj',
             'normal_lucy_tshirt-Slice-54_center_vn.obj',
             'normal_lucy_tshirt-Slice-55_center_vn.obj',
             'normal_lucy_tshirt-Slice-56_center_vn.obj',
             'normal_lucy_tshirt-Slice-57_center_vn.obj',
             'normal_lucy_standard_tee_repaired.obj',
             'normal_lucy_none_repaired.obj',
             'normal_oh_none_repaired.obj'}

for name in filenames:
    filepath = basepath + name
    if os.path.exists(filepath) == False:
        print('Error file does not exist: ' + filepath)
        continue

    reader = objreader.read(filepath)
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

    print ('search result of object: ' + name)
    # test search method
    searchPoint = cloudpoint[50]
    [ind_radius_search, sqdist_radius_search] = octreeNormal.radius_search(searchPoint, 30, 20)
    print ('    radius search: {} results'.format(len(ind_radius_search)))

    # nearest k search
    [ind_k_search, sqdist_k_search] = octreeNormal.nearest_k_search_for_a_point(searchPoint, 20)
    print ('    nearest k search: {} results'.format(len(ind_k_search)))

    # nearest point
    [idx_nearest_search, dst_nearest_search] = octreeNormal.approx_nearest_search(cloudpoint[50]);
    assert idx_nearest_search == 50
    print ('    nearest search: {}'.format(idx_nearest_search))

    # bounding box search
    [bmin, bmax] = octreeNormal.get_bounding_box()
    idx_bounding_search = octreeNormal.box_search(bmin, bmax)
    print ('    bounding box search: {} results'.format(len(idx_bounding_search)))

    # voxel search
    idx_voxel_search = octreeNormal.voxel_search(searchPoint)
    print ('    voxel search: {} results'.format(len(idx_voxel_search)))

#for i in range(0, idx_voxel_search.size):
#    print ('index = ' + str(idx_voxel_search[i]))
#    print ('(' + str(cloudpoint[idx_voxel_search[i]][0]) + ' ' + str(cloudpoint[idx_voxel_search[i]][1]) + ' ' + str(cloudpoint[idx_voxel_search[i]][2]))
