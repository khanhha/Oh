import os.path
import numpy as np
import pcl
import vtk
from objsimple2 import objreader

#
config_kdtree_points_per_leaf = 100
config_kdtree_normal_threshold = True
config_kdtree_normal_threshold_value = 0.8
config_bsasepath = 'G:\\Projects\\Oh\data\\test_data\\'
config_filename = 'normal_lucy_none-Slice-54_center_vn.obj'
#config_filename = 'normal_lucy_none_repaired.obj'
config_color_octree_depth_cube = (0.2, 0.5, 0.3)
#

def pcl_load_point_cloud(basepath, filename):
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
    pcloud = pcl.PointCloud()
    pcloud.from_array(points)
    ncloud = pcl.PointCloud_Normal()
    ncloud.from_array(normals)

    return [pcloud, ncloud]

def pcl_build_kdtree(pcloud, points_per_leaf, normal_split, ncloud, normal_split_thresohld):
    #build tree based on normal and max per leaf node
    kd = pcl.KdTreeFLANN(pcloud)
    kd.set_max_points_per_leaf(points_per_leaf)
    if normal_split:
        kd.set_input_normal_cloud(ncloud)
        kd.set_leaf_normal_threshold(normal_split_thresohld)

    kd.add_points_from_input_cloud()

    return kd

def vtk_build_point_actor(points):
    npoints = len(points)

    vtk_pcoords = vtk.vtkFloatArray()
    vtk_pcoords.SetNumberOfComponents(3)
    vtk_pcoords.SetNumberOfTuples(npoints)
    for i in range(0, npoints):
        p = points[i]
        vtk_pcoords.SetTuple3(i, p[0], p[1], p[2])

    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(vtk_pcoords)

    vertices = vtk.vtkCellArray()
    vertices.Allocate(2 * npoints, 1000)

    for i in range(0, npoints):
        vertices.InsertNextCell(1)
        vertices.InsertCellPoint(i)

    assert vertices.GetNumberOfCells() == npoints

    vtk_poly = vtk.vtkPolyData()
    vtk_poly.SetPoints(vtk_points)
    vtk_poly.SetVerts(vertices)

    # Visualize
    mapper = vtk.vtkPolyDataMapper()
    if vtk.VTK_MAJOR_VERSION <= 5:
        mapper.SetInput(vtk_poly)
    else:
        mapper.SetInputData(vtk_poly)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    return actor

def vtk_build_box_actor(bounds, degenerate_bound = 0.5):
    box_offsets = ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1))
    box_edges = ((0,1), (0,2), (0,3), (4,7), (5,7), (6,7), (1,6), (2,5), (2,4), (3,6), (3,5), (1,4))

    nbox = len(bounds)

    vtk_pcoords = vtk.vtkFloatArray()
    vtk_pcoords.SetNumberOfComponents(3)
    vtk_pcoords.SetNumberOfTuples(nbox * 8)
    delta = [0,0,0]
    for i in range(0, nbox):
        bmin = bounds[i][0]
        bmax = bounds[i][1]
        #delta = (bmax[0] - bmin[0], bmax[1] - bmin[1], bmax[2] - bmin[2])
        for j in range(3):
            delta[j] = bmax[j] - bmin[j]
            delta[j] = max(delta[j], degenerate_bound)

        for j in range(0,8):
            x = bmin[0] + box_offsets[j][0]* delta[0]
            y = bmin[1] + box_offsets[j][1]* delta[1]
            z = bmin[2] + box_offsets[j][2]* delta[2]
            vtk_pcoords.SetTuple3(i*8+j, x, y, z)

    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(vtk_pcoords)

    lines = vtk.vtkCellArray()
    lines.Allocate(nbox * 12 * 2 + nbox * 12, 1000)

    for i in range(0, nbox):
        for e in range(0,12):
            lines.InsertNextCell(2, (i*8 + box_edges[e][0], i*8 + box_edges[e][1]))

    vtk_poly = vtk.vtkPolyData()
    vtk_poly.SetPoints(vtk_points)
    vtk_poly.SetLines(lines)

    ncell = vtk_poly.GetNumberOfCells()

    # Visualize
    mapper = vtk.vtkPolyDataMapper()
    if vtk.VTK_MAJOR_VERSION <= 5:
        mapper.SetInput(vtk_poly)
    else:
        mapper.SetInputData(vtk_poly)
        mapper.Update()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    return actor

def vtk_build_kdtree_leaf_box_actor(kdtree):
    bounds = kdtree.get_all_leaf_nodes_bounding_box()
    return vtk_build_box_actor(bounds)

def vtk_build_point_cloud_actor(cloud):
    npoints = cloud.size
    points = np.ndarray([npoints, 3], dtype=float)
    for i in range(0, npoints):
        p = cloud[i]
        points[i][0] = p[0]
        points[i][1] = p[1]
        points[i][2] = p[2]

    return vtk_build_point_actor(points)

cloud_normal = None
cloud_point = None
kdtree_normal = None

octree_leaf_count = 0
octree_depth = 0

user_depth = 1
user_a_leaf_index = 1

user_view_cloud = True
user_view_a_leaf = False
user_view_all_leaf_center = True
user_view_all_leaf_cube = True
user_view_node_cube_at_max_depth = True
user_view_node_cube_at_depth = False

vtk_renderer = None
vtk_renWin = None
vtk_actor_leaf_cube = None

###########################################################

[cloud_point, cloud_normal] = pcl_load_point_cloud(config_bsasepath, config_filename)
kdtree_normal = pcl_build_kdtree(cloud_point, config_kdtree_points_per_leaf,
                                 config_kdtree_normal_threshold, cloud_normal, config_kdtree_normal_threshold_value)

vtk_actor_leaf_cube = vtk_build_kdtree_leaf_box_actor(kdtree_normal)
vtk_actor_leaf_cube.GetProperty().SetColor(0.9, 0.5, 0.3)
vtk_actor_leaf_cube.GetProperty().SetLineWidth(1)

vtk_actor_cloud_point = vtk_build_point_cloud_actor(cloud_point)
vtk_actor_cloud_point.GetProperty().SetPointSize(2)

camera = vtk.vtkCamera()
camera.SetPosition(1, 1, 1)
camera.SetFocalPoint(0, 0, 0)

vtk_renderer = vtk.vtkRenderer()

vtk_renderer.AddActor(vtk_actor_leaf_cube)
vtk_renderer.AddActor(vtk_actor_cloud_point)

vtk_renderer.SetActiveCamera(camera)
vtk_renderer.ResetCamera()
vtk_renderer.SetBackground(0, 0, 0)

vtk_renWin = vtk.vtkRenderWindow()
vtk_renWin.SetSize(300, 300)
vtk_renWin.AddRenderer(vtk_renderer)

style = vtk.vtkInteractorStyleTrackballCamera()
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(vtk_renWin)
iren.SetInteractorStyle(style)

# interact with data
vtk_renWin.Render()
iren.Start()

# Clean up
del camera
del vtk_renderer
del vtk_renWin
del iren


