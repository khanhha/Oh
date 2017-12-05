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

def vtk_build_octree_leaf_center_actor(octree):
    [keys, depths] = octreeNormal.get_all_leaf_keys()
    nleaf = len(keys)
    points = np.ndarray([nleaf, 3], dtype=float)
    for k in range(nleaf):
        c = octree.gen_voxel_center_from_octree_key(keys[k], depths[k])
        points[k][0] = c[0]
        points[k][1] = c[1]
        points[k][2] = c[2]

    return vtk_build_point_actor(points)

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
octreeNormal = pcl.OctreePointCloudNormal(0.1)
octreeNormal.set_input_cloud(cloudpoint)
octreeNormal.set_input_normal_cloud(cloudnormal)
octreeNormal.enable_dynamic_depth(100)
octreeNormal.set_leaf_normal_threshold(0.8)
octreeNormal.add_points_from_input_cloud()

vtk_leaf_center_actor = vtk_build_octree_leaf_center_actor(octreeNormal)
vtk_leaf_center_actor.GetProperty().SetColor(0.2, 0.63, 0.79)
vtk_leaf_center_actor.GetProperty().SetPointSize(2)

vtk_cloud_point_actor = vtk_build_point_cloud_actor(cloudpoint)

camera = vtk.vtkCamera()
camera.SetPosition(1, 1, 1)
camera.SetFocalPoint(0, 0, 0)

renderer = vtk.vtkRenderer()
renderer.AddActor(vtk_leaf_center_actor)
renderer.AddActor(vtk_cloud_point_actor)
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


