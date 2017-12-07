import os.path
import numpy as np
import pcl
import vtk
from objsimple2 import objreader

#
config_octree_resolution = 10
config_octree_points_per_leaf = 100
config_octree_dynamic_leaf = True
config_octree_normal_threshold = True
config_octree_normal_threshold_value = 0.8
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

def pcl_build_octree(pcloud, res, dynamic, points_per_leaf, normal_split, ncloud, normal_split_thresohld):
    #build tree based on normal and max per leaf node
    ocnormal = pcl.OctreePointCloudNormal(res)
    ocnormal.set_input_cloud(pcloud)

    if dynamic:
        ocnormal.enable_dynamic_depth(points_per_leaf)
        if normal_split:
            ocnormal.set_input_normal_cloud(ncloud)
            ocnormal.set_leaf_normal_threshold(normal_split_thresohld)

    ocnormal.add_points_from_input_cloud()

    return ocnormal

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

def vtk_build_box_actor(bounds):
    box_offsets = ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1))
    box_edges = ((0,1), (0,2), (0,3), (4,7), (5,7), (6,7), (1,6), (2,5), (2,4), (3,6), (3,5), (1,4))

    nbox = len(bounds)

    vtk_pcoords = vtk.vtkFloatArray()
    vtk_pcoords.SetNumberOfComponents(3)
    vtk_pcoords.SetNumberOfTuples(nbox * 8)
    for i in range(0, nbox):
        bmin = bounds[i][0]
        bmax = bounds[i][1]
        delta = (bmax[0] - bmin[0], bmax[1] - bmin[1], bmax[2] - bmin[2])
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

def vtk_build_octree_leaf_center_actor(octree):
    [keys, depths] = octree.get_all_leaf_keys()
    nleaf = len(keys)
    points = np.ndarray([nleaf, 3], dtype=float)
    for k in range(nleaf):
        c = octree.gen_voxel_center_from_octree_key(keys[k], depths[k])
        for i in range(0,3):
            points[k][i] = c[i]

    return vtk_build_point_actor(points)

def vtk_build_octree_leaf_box_actor(octree):
    [keys, depths] = octree.get_all_leaf_keys()
    nleaf = len(keys)
    bounds = np.ndarray([nleaf, 2, 3], dtype = float)
    for k in range(nleaf):
        [bmin, bmax] = octree.gen_voxel_bounds_from_octree_key(keys[k], depths[k])
        for i in range(0,3):
            bounds[k][0][i] = bmin[i]
            bounds[k][1][i] = bmax[i]

    return vtk_build_box_actor(bounds)

def vtk_build_octree_box_at_max_depth_actor(octree, depth, fromRootToDepth):
    if(fromRootToDepth):
        [keys, depths] = octree.gell_all_node_keys_at_max_depth(depth)
    else:
        [keys, depths] = octree.gell_all_node_keys_at_depth(depth)

    nleaf = len(keys)
    bounds = np.ndarray([nleaf, 2, 3], dtype = float)
    for k in range(nleaf):
        [bmin, bmax] = octree.gen_voxel_bounds_from_octree_key(keys[k], depths[k])
        for i in range(0,3):
            bounds[k][0][i] = bmin[i]
            bounds[k][1][i] = bmax[i]

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

cloudnormal = None
cloudpoint = None
octreeNormal = None

user_view_cloud = True
user_view_all_leaf_center = True
user_view_all_leaf_cube = True
user_view_depth = 1
user_view_node_cube_at_max_depth = True
user_view_node_cube_at_depth = False

vtk_renderer = None
vtk_renWin = None
vtk_actor_leaf_center = None
vtk_actor_leaf_cube = None
vtk_actor_max_depth_cube = None
vtk_actor_cloud_point = None
vtk_actor_text = None

def vtk_update_max_depth_cube():
    global vtk_actor_max_depth_cube
    vtk_renderer.RemoveActor(vtk_actor_max_depth_cube)
    if user_view_node_cube_at_max_depth:
        vtk_actor_max_depth_cube = vtk_build_octree_box_at_max_depth_actor(octreeNormal, user_view_depth, True)
    elif user_view_node_cube_at_depth:
        vtk_actor_max_depth_cube = vtk_build_octree_box_at_max_depth_actor(octreeNormal, user_view_depth, False)

    vtk_actor_max_depth_cube.GetProperty().SetColor(config_color_octree_depth_cube[0],
                                                    config_color_octree_depth_cube[1],
                                                    config_color_octree_depth_cube[2])
    vtk_renderer.AddActor(vtk_actor_max_depth_cube)

def vtk_text_update():
    global vtk_actor_text
    if vtk_actor_text is not None:
        vtk_renderer.RemoveActor(vtk_actor_text)

    text = ('   press 1 : view point cloud:  ' + str(user_view_cloud) + '\n' +
            '   press 2 : view all leaf centers:  ' + str(user_view_all_leaf_center) + '\n' +
            '   press 3 : view all leaf cubes:  ' + str(user_view_all_leaf_cube) + '\n' +
            '   press 4 : view all node cubes from root to depth value: ' + str(user_view_node_cube_at_max_depth) + '\n' +
            '   press 5 : view all node cubes at depth value:  ' + str(user_view_node_cube_at_depth) + '\n' +
            '   press 8 : decrease current depth value: ' + str(user_view_depth) + '\n' +
            '   press 9 : increase current depth value: ' + str(user_view_depth) + '\n' +
            ' ' + '\n')

    vtk_actor_text = vtk.vtkTextActor()
    vtk_actor_text.SetInput(text)
    vtk_actor_text.GetTextProperty().SetColor((0, 1, 1))
    vtk_renderer.AddActor(vtk_actor_text)

def vtk_ev_keypressed_handle(obj, event):
    global vtk_renderer
    global user_view_cloud, user_view_all_leaf_center, user_view_all_leaf_cube, user_view_depth, user_view_node_cube_at_max_depth, user_view_node_cube_at_depth
    key = obj.GetKeySym()
    if key == '1':
        user_view_cloud = not user_view_cloud
        if(user_view_cloud):
            vtk_actor_cloud_point.VisibilityOn()
        else:
            vtk_actor_cloud_point.VisibilityOff()
    elif key == '2':
        user_view_all_leaf_center = not user_view_all_leaf_center
        if user_view_all_leaf_center:
            vtk_actor_leaf_center.VisibilityOn()
        else:
            vtk_actor_leaf_center.VisibilityOff()
    elif key == '3':
        user_view_all_leaf_cube = not user_view_all_leaf_cube
        if user_view_all_leaf_cube:
            vtk_actor_leaf_cube.VisibilityOn()
        else:
            vtk_actor_leaf_cube.VisibilityOff()
    elif key == '4':
        user_view_node_cube_at_max_depth = not user_view_node_cube_at_max_depth
        if user_view_node_cube_at_max_depth and user_view_node_cube_at_depth:
            user_view_node_cube_at_depth = False

        vtk_update_max_depth_cube()

        if user_view_node_cube_at_max_depth:
            vtk_actor_max_depth_cube.VisibilityOn()
        else:
            vtk_actor_max_depth_cube.VisibilityOff()
    elif key == '5':
        user_view_node_cube_at_depth = not user_view_node_cube_at_depth
        if user_view_node_cube_at_max_depth and user_view_node_cube_at_depth:
            user_view_node_cube_at_max_depth = False

        vtk_update_max_depth_cube()

        if user_view_node_cube_at_depth:
            vtk_actor_max_depth_cube.VisibilityOn()
        else:
            vtk_actor_max_depth_cube.VisibilityOff()
    elif key == '9':
        user_view_depth += 1
        vtk_update_max_depth_cube()
    elif key == '8':
        user_view_depth -= 1
        user_view_depth = max(user_view_depth, 1)
        vtk_update_max_depth_cube()

    vtk_text_update()
    vtk_renderer.Render()
    vtk_renWin.Render()

###########################################################
basepath = 'G:\\Projects\\Oh\data\\test_data\\'
filename = 'normal_lucy_none-Slice-54_center_vn.obj'

[cloudpoint, cloudnormal] = pcl_load_point_cloud(basepath, filename)
octreeNormal = pcl_build_octree(cloudpoint, config_octree_resolution,
                                config_octree_dynamic_leaf, config_octree_points_per_leaf,
                                config_octree_normal_threshold, cloudnormal, config_octree_normal_threshold_value)

vtk_actor_leaf_center = vtk_build_octree_leaf_center_actor(octreeNormal)
vtk_actor_leaf_center.GetProperty().SetColor(0.9, 0.2, 0.1)
vtk_actor_leaf_center.GetProperty().SetPointSize(4)

vtk_actor_leaf_cube = vtk_build_octree_leaf_box_actor(octreeNormal)
vtk_actor_leaf_cube.GetProperty().SetColor(0.9, 0.5, 0.3)
vtk_actor_leaf_cube.GetProperty().SetLineWidth(1)

vtk_actor_max_depth_cube = vtk_build_octree_box_at_max_depth_actor(octreeNormal, 1, True)
vtk_actor_max_depth_cube.GetProperty().SetColor(config_color_octree_depth_cube[0],config_color_octree_depth_cube[1], config_color_octree_depth_cube[2])
vtk_actor_max_depth_cube.GetProperty().SetLineWidth(1)

vtk_actor_cloud_point = vtk_build_point_cloud_actor(cloudpoint)
vtk_actor_cloud_point.GetProperty().SetPointSize(2)

camera = vtk.vtkCamera()
camera.SetPosition(1, 1, 1)
camera.SetFocalPoint(0, 0, 0)

vtk_renderer = vtk.vtkRenderer()
vtk_renderer.AddActor(vtk_actor_leaf_center)
vtk_renderer.AddActor(vtk_actor_cloud_point)
vtk_renderer.AddActor(vtk_actor_leaf_cube)
vtk_renderer.AddActor(vtk_actor_max_depth_cube)
vtk_renderer.SetActiveCamera(camera)
vtk_renderer.ResetCamera()
vtk_renderer.SetBackground(0, 0, 0)

vtk_text_update()

vtk_renWin = vtk.vtkRenderWindow()
vtk_renWin.SetSize(300, 300)
vtk_renWin.AddRenderer(vtk_renderer)

style = vtk.vtkInteractorStyleTrackballCamera()
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(vtk_renWin)
iren.SetInteractorStyle(style)
iren.AddObserver("KeyPressEvent", vtk_ev_keypressed_handle)

# interact with data
vtk_renWin.Render()
iren.Start()

# Clean up
del camera
del vtk_renderer
del vtk_renWin
del iren


