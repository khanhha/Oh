import os.path
import numpy as np
import pcl
import vtk
from objsimple2 import objreader

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
    [keys, depths] = octreeNormal.get_all_leaf_keys()
    nleaf = len(keys)
    points = np.ndarray([nleaf, 3], dtype=float)
    for k in range(nleaf):
        c = octree.gen_voxel_center_from_octree_key(keys[k], depths[k])
        for i in range(0,3):
            points[k][i] = c[i]

    return vtk_build_point_actor(points)

def vtk_build_octree_leaf_box_actor(octree):
    [keys, depths] = octreeNormal.get_all_leaf_keys()
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
octreeNormal = pcl.OctreePointCloudNormal(10)
octreeNormal.set_input_cloud(cloudpoint)
octreeNormal.set_input_normal_cloud(cloudnormal)
octreeNormal.set_leaf_normal_threshold(0.8)
octreeNormal.enable_dynamic_depth(100)
octreeNormal.add_points_from_input_cloud()

vtk_leaf_center_actor = vtk_build_octree_leaf_center_actor(octreeNormal)
vtk_leaf_center_actor.GetProperty().SetColor(0.2, 0.63, 0.79)
vtk_leaf_center_actor.GetProperty().SetPointSize(2)

vtk_leaf_box_actor = vtk_build_octree_leaf_box_actor(octreeNormal)
vtk_leaf_box_actor.GetProperty().SetColor(0.9, 0.5, 0.3)
vtk_leaf_box_actor.GetProperty().SetLineWidth(1)

vtk_cloud_point_actor = vtk_build_point_cloud_actor(cloudpoint)
vtk_cloud_point_actor.GetProperty().SetPointSize(3)

camera = vtk.vtkCamera()
camera.SetPosition(1, 1, 1)
camera.SetFocalPoint(0, 0, 0)

renderer = vtk.vtkRenderer()
#renderer.AddActor(vtk_leaf_center_actor)
renderer.AddActor(vtk_cloud_point_actor)
renderer.AddActor(vtk_leaf_box_actor)
renderer.SetActiveCamera(camera)
renderer.ResetCamera()
renderer.SetBackground(0, 0, 0)

renWin = vtk.vtkRenderWindow()
renWin.AddRenderer(renderer)

style = vtk.vtkInteractorStyleTrackballCamera()
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)
iren.SetInteractorStyle(style)

renWin.SetSize(300, 300)

# interact with data
renWin.Render()
iren.Start()

# Clean up
del camera
del renderer
del renWin
del iren


