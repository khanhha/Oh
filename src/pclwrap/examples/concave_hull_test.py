import os.path
import numpy as np
import pcl
import vtk
import sys
from. import vtkDrawUtil
from objsimple2 import objreader, objwriter


def vtk_build_continuous_segments_actor(points, witdh = 2.0):
    npoints = len(points)

    vtk_pcoords = vtk.vtkFloatArray()
    vtk_pcoords.SetNumberOfComponents(3)
    vtk_pcoords.SetNumberOfTuples(npoints)
    for i in range(0, npoints):
        p = points[i]
        vtk_pcoords.SetTuple3(i, p[0], p[1], p[2])

    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(vtk_pcoords)

    lines = vtk.vtkCellArray()
    lines.Allocate(4 * npoints, 1000)

    for i in range(0, npoints):
        lines.InsertNextCell(2)
        lines.InsertCellPoint(i)
        lines.InsertCellPoint((i+1)%npoints)

    assert lines.GetNumberOfCells() == npoints

    vtk_poly = vtk.vtkPolyData()
    vtk_poly.SetPoints(vtk_points)
    vtk_poly.SetLines(lines)

    # Visualize
    mapper = vtk.vtkPolyDataMapper()
    if vtk.VTK_MAJOR_VERSION <= 5:
        mapper.SetInput(vtk_poly)
    else:
        mapper.SetInputData(vtk_poly)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    actor.GetProperty().SetLineWidth(witdh)

    return actor

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

basepath = 'D:\\Projects\\Oh\data\\test_data\\'
filenames = {#'lucy_none-Slice-54_center_vn.obj',
             #'lucy_none-Slice-55_center_vn.obj',
             #'lucy_none-Slice-56_center_vn.obj',
             #'lucy_none-Slice-57_center_vn.obj',
             # 'lucy_tshirt-Slice-54_center_vn.obj',
             # 'lucy_tshirt-Slice-55_center_vn.obj',
             # 'lucy_tshirt-Slice-56_center_vn.obj',
            'lucy_tshirt-Slice-57_center_vn.obj',
            }

#for name in filenames:
name = 'normal_lucy_none-Slice-54_center_vn.obj'
filepath = basepath + name
if os.path.exists(filepath) == False:
    print('Error file does not exist: ' + filepath)

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

bmin = [sys.float_info.max, sys.float_info.max, sys.float_info.max]
bmax = [-sys.float_info.max, -sys.float_info.max, -sys.float_info.max]
npoints = cloudpoint.size
for i in range(npoints):
    p = cloudpoint[i]
    for d in range(3):
        bmin[d] = min(p[d], bmin[d])
        bmax[d] = max(p[d], bmax[d])

bsize = [bmax[0] - bmin[0], bmax[1] - bmin[1], bmax[2] - bmin[2]]
mindim = 0
if bsize[1] < bsize[mindim]:
    mindim = 1
if bsize[2] < bsize[mindim]:
    mindim = 2

radius = 2
plane_dim_val = 0.5 * (bmin[mindim] + bmax[mindim])
near_plane = []
for i in range(npoints):
    p = cloudpoint[i]
    if abs(p[mindim] - plane_dim_val) < radius :
        near_plane.append(p)

near_plane_cloud  = pcl.PointCloud()
near_plane_cloud.from_list(near_plane)

hbuilder = pcl.ConcaveHull()
hbuilder.set_input_cloud(near_plane_cloud)
hbuilder.set_dimension(2)
hbuilder.set_alpha(20)
result , contours, perimeters = hbuilder.reconstruct_contours_perimeters()

camera = vtk.vtkCamera()
camera.SetPosition(1, 1, 1)
camera.SetFocalPoint(0, 0, 0)

hull_arr = result.to_array()
nhull_point = len(hull_arr)

avg_dim_val = 0.0
dim = 2
if dim != -1:
    for i in range(nhull_point):
        avg_dim_val += hull_arr[i][dim]

    avg_dim_val /= nhull_point

    for i in range(nhull_point):
        hull_arr[i][dim] = avg_dim_val

vtk_org_cloud_actor = vtk_build_point_actor(near_plane_cloud.to_array())
vtk_convex_hull_cloud_actor = vtk_build_point_actor(result.to_array()) #vtk_build_continuous_segments_actor(hull_arr)
vtk_convex_hull_cloud_actor.GetProperty().SetColor(0.9, 0.9, 0.0)
vtk_convex_hull_cloud_actor.GetProperty().SetPointSize(3)

vtk_renderer = vtk.vtkRenderer()
vtk_renderer.AddActor(vtk_org_cloud_actor)
vtk_renderer.AddActor(vtk_convex_hull_cloud_actor)

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