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
    kdtree = pcl.KdTreeFLANN(pcloud)

    if normal_split:
        kdtree.set_input_normal_cloud(ncloud)
        kdtree.set_leaf_normal_threshold(normal_split_thresohld)

    kdtree.add_points_from_input_cloud()

    return kdtree

def pcl_kdtree_infor_string(kdtree, cloud):
    npoints = cloud.size
    depth = kdtree.get_max_depth()
    nleaf = kdtree.get_leaf_count()
    nbranch = kdtree.get_branch_count()

    info = ('\n' +
            '   number of points: ' + str(npoints) +'\n'
            '   points per leaf threshold: ' + str(config_kdtree_points_per_leaf) + '\n' +
            '   enable normal threshold: ' + str(config_kdtree_normal_threshold) +'\n' +
            '   normal threshold: ' + str(config_kdtree_normal_threshold_value) +'\n' +
            '   maxmimum tree depth: ' + str(depth) + '\n' +
            '   number of leafs: ' + str(nleaf) +'\n' +
            '   number of branches: ' + str(nbranch) + '\n\n'
            )

    return info

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

def vtk_build_box_actor(bounds, degenerate_threshold = 0.5):
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
        for j in range(3):
            delta[j] = max(bmax[j] - bmin[j], degenerate_threshold)
            #delta = (bmax[0] - bmin[0], bmax[1] - bmin[1], bmax[2] - bmin[2])
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

def vtk_build_kdtree_leaf_center_actor(kdtree):
    [bounds, nleaf] = kdtree.get_all_leaf_nodes_bounding_box()
    points = np.ndarray([nleaf, 3], dtype=float)
    for i in range(nleaf):
        for k in range(3):
            points[i][k] = 0.5 * (bounds[i][0][k] + bounds[i][1][k])

    return vtk_build_point_actor(points)

def vtk_build_kdtree_leaf_box_actor(kdtree):
    [bounds, nleaf] = kdtree.get_all_leaf_nodes_bounding_box()
    return vtk_build_box_actor(bounds)

def vtk_build_octree_a_leaf_points_actor(kdtree, pcloud, leaf_center):
    vindices = kdtree.get_leaf_point_indices(leaf_center)
    nvertices = len(vindices)
    if nvertices == 0:
        print('Cloud not find points at leaf center: ( {}, {}, {} )'.format(leaf_center[0], leaf_center[1], leaf_center[2]))
    points = np.ndarray([nvertices, 3], dtype=float)
    for i in range(nvertices):
        p = pcloud[vindices[i]]
        for k in range(0, 3):
            points[i][k] = p[k]

    actor = vtk_build_point_actor(points)
    actor.GetProperty().SetColor(0.9, 0.9, 0.2)
    actor.GetProperty().SetPointSize(4)
    return actor


def vtk_build_octree_box_at_max_depth_actor(kdtree, depth, fromRootToDepth):
    if(fromRootToDepth):
        [bounds, depths]    = kdtree.get_nodes_bounding_box_at_max_depth(depth)
    else:
        bounds              = kdtree.get_nodes_bounding_box_at_depth(depth)

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
kdtree_leaf_count = 0
kdtree_depth = 0
kdtree_leaf_centers = None

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
vtk_actor_leaf_center = None
vtk_actor_leaf_cube = None
vtk_actor_a_leaf = None
vtk_actor_max_depth_cube = None
vtk_actor_cloud_point = None
vtk_actor_text = None

def kdtree_calculate_leaf_centers(kdtree):
    global kdtree_leaf_centers
    [bounds, nleaf] = kdtree.get_all_leaf_nodes_bounding_box()
    kdtree_leaf_centers = np.ndarray([nleaf, 3], dtype=float)
    for i in range(nleaf):
        for k in range(3):
            kdtree_leaf_centers[i][k] = 0.5 * (bounds[i][0][k] + bounds[i][1][k])

def vtk_update_actor_max_depth_cube():
    global vtk_actor_max_depth_cube
    vtk_renderer.RemoveActor(vtk_actor_max_depth_cube)
    if user_view_node_cube_at_max_depth:
        vtk_actor_max_depth_cube = vtk_build_octree_box_at_max_depth_actor(kdtree_normal, user_depth, True)
    elif user_view_node_cube_at_depth:
        vtk_actor_max_depth_cube = vtk_build_octree_box_at_max_depth_actor(kdtree_normal, user_depth, False)

    vtk_actor_max_depth_cube.GetProperty().SetColor(config_color_octree_depth_cube[0],
                                                    config_color_octree_depth_cube[1],
                                                    config_color_octree_depth_cube[2])
    vtk_renderer.AddActor(vtk_actor_max_depth_cube)

def vtk_update_actor_a_leaf():
    global vtk_actor_a_leaf
    vtk_renderer.RemoveActor(vtk_actor_a_leaf)
    vtk_actor_a_leaf = vtk_build_octree_a_leaf_points_actor(kdtree_normal, cloud_point, kdtree_leaf_centers[user_a_leaf_index])
    vtk_renderer.AddActor(vtk_actor_a_leaf)
    if user_view_a_leaf:
        vtk_actor_a_leaf.VisibilityOn()
    else:
        vtk_actor_a_leaf.VisibilityOff()

def vtk_text_update():
    global vtk_actor_text
    if vtk_actor_text is not None:
        vtk_renderer.RemoveActor(vtk_actor_text)

    text = ('   press 1 : view point cloud:  ' + str(user_view_cloud) + '\n' +
            '\n' +
            '   press 2 : view points of a leaf: ' + str(user_view_a_leaf)+ '  ' + str(user_a_leaf_index) + '\n' +
            '   press F8: increase view leaf index: ' + str(user_a_leaf_index) + '\n' +
            '   press F9: decrease view leaf index: ' + str(user_a_leaf_index) + '\n' +
            '\n'+
            '   press 3 : view all leaf centers:  ' + str(user_view_all_leaf_center) + '\n' +
            '   press 4 : view all leaf cubes:  ' + str(user_view_all_leaf_cube) + '\n' +
            '\n' +
            '   press 5 : view all node cubes from root to depth value: ' + str(user_view_node_cube_at_max_depth) + ' ' + str(user_depth) +'\n' +
            '   press 6 : view all node cubes at depth value:  ' + str(user_view_node_cube_at_depth) + ' ' + str(user_depth) + '\n' +
            '   press 8 : decrease current depth value: ' + str(user_depth) + '\n' +
            '   press 9 : increase current depth value: ' + str(user_depth) + '\n' +
            ' ' + '\n')

    text += pcl_kdtree_infor_string(kdtree_normal, cloud_point)

    vtk_actor_text = vtk.vtkTextActor()
    vtk_actor_text.SetInput(text)
    vtk_actor_text.GetTextProperty().SetColor((0, 1, 1))
    vtk_renderer.AddActor(vtk_actor_text)

def vtk_ev_keypressed_handle(obj, event):
    global vtk_renderer
    global user_view_cloud, user_view_all_leaf_center, user_view_all_leaf_cube, user_depth, user_view_node_cube_at_max_depth, user_view_node_cube_at_depth, user_a_leaf_index, user_view_a_leaf
    key = obj.GetKeySym()

    if key == '1':
        user_view_cloud = not user_view_cloud
        if(user_view_cloud):
            vtk_actor_cloud_point.VisibilityOn()
        else:
            vtk_actor_cloud_point.VisibilityOff()
    elif key == '2':
       user_view_a_leaf = not user_view_a_leaf
       user_view_cloud = False
       vtk_actor_cloud_point.VisibilityOff()
       if user_view_a_leaf:
           vtk_actor_a_leaf.VisibilityOn()
       else:
           vtk_actor_a_leaf.VisibilityOff()

    elif key == '3':
        user_view_all_leaf_center = not user_view_all_leaf_center
        if user_view_all_leaf_center:
            vtk_actor_leaf_center.VisibilityOn()
        else:
            vtk_actor_leaf_center.VisibilityOff()

    elif key == '4':
        user_view_all_leaf_cube = not user_view_all_leaf_cube
        if user_view_all_leaf_cube:
            vtk_actor_leaf_cube.VisibilityOn()
        else:
            vtk_actor_leaf_cube.VisibilityOff()

    elif key == '5':
        user_view_node_cube_at_max_depth = not user_view_node_cube_at_max_depth
        if user_view_node_cube_at_max_depth and user_view_node_cube_at_depth:
            user_view_node_cube_at_depth = False

        vtk_update_actor_max_depth_cube()

        if user_view_node_cube_at_max_depth:
            vtk_actor_max_depth_cube.VisibilityOn()
        else:
            vtk_actor_max_depth_cube.VisibilityOff()

    elif key == '6':
        user_view_node_cube_at_depth = not user_view_node_cube_at_depth
        if user_view_node_cube_at_max_depth and user_view_node_cube_at_depth:
            user_view_node_cube_at_max_depth = False

        vtk_update_actor_max_depth_cube()

        if user_view_node_cube_at_depth:
            vtk_actor_max_depth_cube.VisibilityOn()
        else:
            vtk_actor_max_depth_cube.VisibilityOff()

    elif key == '9':
        user_depth += 1
        user_depth = min(user_depth, kdtree_depth)
        vtk_update_actor_max_depth_cube()
    elif key == '8':
        user_depth -= 1
        user_depth = max(user_depth, 0)
        vtk_update_actor_max_depth_cube()
    elif key == "F9":
        user_a_leaf_index += 1
        user_a_leaf_index = min(user_a_leaf_index, kdtree_leaf_count - 1)
        vtk_update_actor_a_leaf()
    elif key == "F8":
        user_a_leaf_index -=1
        user_a_leaf_index = max(user_a_leaf_index, 0)
        vtk_update_actor_a_leaf()

    vtk_text_update()
    vtk_renderer.Render()
    vtk_renWin.Render()

###########################################################

[cloud_point, cloud_normal] = pcl_load_point_cloud(config_bsasepath, config_filename)
kdtree_normal = pcl_build_kdtree(cloud_point,
                                 config_kdtree_points_per_leaf,
                                 config_kdtree_normal_threshold, cloud_normal, config_kdtree_normal_threshold_value)
kdtree_leaf_count = kdtree_normal.get_leaf_count()
kdtree_depth = kdtree_normal.get_max_depth()

kdtree_calculate_leaf_centers(kdtree_normal)

vtk_actor_leaf_center = vtk_build_kdtree_leaf_center_actor(kdtree_normal)
vtk_actor_leaf_center.GetProperty().SetColor(0.9, 0.2, 0.1)
vtk_actor_leaf_center.GetProperty().SetPointSize(4)

vtk_actor_leaf_cube = vtk_build_kdtree_leaf_box_actor(kdtree_normal)
vtk_actor_leaf_cube.GetProperty().SetColor(0.9, 0.5, 0.3)
vtk_actor_leaf_cube.GetProperty().SetLineWidth(1)

vtk_actor_a_leaf = vtk_build_octree_a_leaf_points_actor(kdtree_normal, cloud_point, kdtree_leaf_centers[0])

vtk_actor_max_depth_cube = vtk_build_octree_box_at_max_depth_actor(kdtree_normal, 1, True)
vtk_actor_max_depth_cube.GetProperty().SetColor(config_color_octree_depth_cube[0],config_color_octree_depth_cube[1], config_color_octree_depth_cube[2])
vtk_actor_max_depth_cube.GetProperty().SetLineWidth(1)

vtk_actor_cloud_point = vtk_build_point_cloud_actor(cloud_point)
vtk_actor_cloud_point.GetProperty().SetPointSize(2)

camera = vtk.vtkCamera()
camera.SetPosition(1, 1, 1)
camera.SetFocalPoint(0, 0, 0)

vtk_renderer = vtk.vtkRenderer()

vtk_renderer.AddActor(vtk_actor_leaf_center)
vtk_renderer.AddActor(vtk_actor_cloud_point)
vtk_renderer.AddActor(vtk_actor_leaf_cube)
vtk_renderer.AddActor(vtk_actor_max_depth_cube)
vtk_renderer.AddActor(vtk_actor_a_leaf)

vtk_renderer.SetActiveCamera(camera)
vtk_renderer.ResetCamera()
vtk_renderer.SetBackground(0, 0, 0)

vtk_text_update()
vtk_update_actor_a_leaf()
vtk_update_actor_max_depth_cube()

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


