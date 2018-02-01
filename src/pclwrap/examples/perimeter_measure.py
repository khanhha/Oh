import os.path
import numpy as np
import pcl
import vtk
import sys
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

def min_dimension(bmin, bmax):
    bsize = [bmax[0] - bmin[0], bmax[1] - bmin[1], bmax[2] - bmin[2]]
    mindim = 0
    if bsize[1] < bsize[mindim]:
        mindim = 1
    if bsize[2] < bsize[mindim]:
        mindim = 2
    return mindim

def centroid(cloud, vindices):
    vindices = oc.voxel_search(c)
    nvertices = len(vindices)
    p = [0, 0, 0]
    for i in range(nvertices):
        tmp = cloudpoint[vindices[i]]
        p[0] = p[0] + tmp[0]
        p[1] = p[1] + tmp[1]
        p[2] = p[2] + tmp[2]

    p[0] /= nvertices
    p[1] /= nvertices
    p[2] /= nvertices

    return p


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

oc = pcl.OctreePointCloudSearch(4)
oc.set_input_cloud(cloudpoint)
oc.add_points_from_input_cloud()
[bmin, bmax] = cloudpoint.calculate_bounding_box()
mindim = min_dimension(bmin, bmax)
mid = 0.5 * (bmin[mindim] + bmax[mindim])

slice_points = []

[keys, depths] = oc.get_all_leaf_keys()
nleaf = len(keys)
for leaf_idx in range(nleaf):
    lmin, lmax = oc.gen_voxel_bounds_from_octree_key(keys[leaf_idx], depths[leaf_idx])
    if lmin[mindim] < mid and mid < lmax[mindim]:
        c = oc.gen_voxel_center_from_octree_key(keys[leaf_idx], depths[leaf_idx])
        vindices = oc.voxel_search(c)
        slice_points.append(centroid(cloudpoint, vindices))

slice_cloud = pcl.PointCloud()
slice_cloud.from_list(slice_points)

hbuilder = pcl.ConcaveHull()
hbuilder.set_input_cloud(slice_cloud)
hbuilder.set_dimension(2)
hbuilder.set_alpha(10)
result , contours, perimeters = hbuilder.reconstruct_contours_perimeters()

camera = vtk.vtkCamera()
camera.SetPosition(1, 1, 1)
camera.SetFocalPoint(0, 0, 0)

vtk_org_cloud_actor = vtk_build_point_actor(slice_cloud.to_array())
vtk_convex_hull_cloud_actor = vtk_build_point_actor(result.to_array())
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