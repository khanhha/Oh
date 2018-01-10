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

basepath = 'G:\\Projects\\Oh\data\\test_data\\'
filenames = {'normal_lucy_none-Slice-54_center_vn.obj'
             #,
             # 'normal_lucy_none-Slice-55_center_vn.obj',
             # 'normal_lucy_none-Slice-56_center_vn.obj',
             # 'normal_lucy_none-Slice-57_center_vn.obj',
             # 'normal_lucy_tshirt-Slice-54_center_vn.obj',
             # 'normal_lucy_tshirt-Slice-55_center_vn.obj',
             # 'normal_lucy_tshirt-Slice-56_center_vn.obj',
             # 'normal_lucy_tshirt-Slice-57_center_vn.obj',
             # 'normal_lucy_standard_tee_repaired.obj',
             # 'normal_lucy_none_repaired.obj',
             # 'normal_oh_none_repaired.obj'
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
cloudnormal = pcl.PointCloud_Normal()
cloudnormal.from_array(normals)

# build tree based on normal and max per leaf node
octreeNormal = pcl.OctreePointCloudNormal(0.1)
octreeNormal.set_input_cloud(cloudpoint)
octreeNormal.set_input_normal_cloud(cloudnormal)
octreeNormal.enable_dynamic_depth(100)
octreeNormal.set_leaf_normal_threshold(0.8)
octreeNormal.add_points_from_input_cloud()

sampler = pcl.OctreeSampling()
sampler.set_input_cloud(cloudpoint)
sampler.set_sampling_resolution(5)
sampler.set_resample_method(pcl.ResampleMethod.UNIFORM)
sampler.set_interpolation_method(pcl.InterpolationMethod.HEIGHT_INTERPOLATION)
result = sampler.filter()
camera = vtk.vtkCamera()
camera.SetPosition(1, 1, 1)
camera.SetFocalPoint(0, 0, 0)

vtk_org_cloud_actor = vtk_build_point_actor(cloudpoint.to_array())
vtk_sampled_cloud_actor = vtk_build_point_actor(result.to_array())
vtk_sampled_cloud_actor.GetProperty().SetColor(0.9, 0.9, 0.0)
vtk_sampled_cloud_actor.GetProperty().SetPointSize(3)

vtk_renderer = vtk.vtkRenderer()
vtk_renderer.AddActor(vtk_org_cloud_actor)
vtk_renderer.AddActor(vtk_sampled_cloud_actor)

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