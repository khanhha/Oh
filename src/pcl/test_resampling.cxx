/*=========================================================================
Program:   Visualization Toolkit
Module:    Cone.cxx
Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
All rights reserved.
See Copyright.txt or http://www.kitware.com/Copyright.htm for details.
This software is distributed WITHOUT ANY WARRANTY; without even
the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE.  See the above copyright notice for more information.
=========================================================================*/
//
// This example creates a polygonal model of a cone, and then renders it to
// the screen. It will rotate the cone 360 degrees and then exit. The basic
// setup of source -> mapper -> actor -> renderer -> renderwindow is
// typical of most VTK programs.
//

// First include the required header files for the VTK classes we are using.
#include <vtkAutoInit.h>
VTK_MODULE_INIT(vtkRenderingOpenGL);
#include <vtkVersion.h>
#include <vtkSmartPointer.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkPointData.h>
#include <vtkCellArray.h>
#include <vtkFloatArray.h>
#include <vtkUnsignedCharArray.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkVertexGlyphFilter.h>
#include <vtkProperty.h>
#include "vtkInteractorStyleTrackballCamera.h"
// For compatibility with new VTK generic data arrays
#ifdef vtkGenericDataArray_h
#define InsertNextTupleValue InsertNextTypedTuple
#endif

#include <string>
#include <pcl/point_cloud.h>
#include <pcl/io/obj_loader.h>
#include <pcl/filters/uniform_sampling.h>
#include <pcl/filters/uniform_octree_sampling.h>

#include <Eigen/Dense>

using namespace  std;
using namespace  Eigen;
using namespace  pcl;

void test_uniform_sample()
{
	string filename = "normal_lucy_none-Slice-54_center_vn.obj";
	string basepath = "G:\\Projects\\Oh\\data\\test_data\\";
	PointCloud<PointXYZ>::Ptr cloud = PointCloud<PointXYZ>::Ptr(new PointCloud<PointXYZ>());
	PointCloud<PointXYZ>::Ptr out_cloud = PointCloud<PointXYZ>::Ptr(new PointCloud<PointXYZ>());

	io::cloud_load_point_cloud(basepath + filename, basepath, cloud);

	UniformSampling<PointXYZ> sampler;
	sampler.setInputCloud(cloud);
	sampler.setRadiusSearch(1);
	sampler.filter(*out_cloud);
}

void test_uniform_octree_sample()
{
	string filename = "normal_lucy_none-Slice-54_center_vn.obj";
	string basepath = "G:\\Projects\\Oh\\data\\test_data\\";
	PointCloud<PointXYZ>::Ptr cloud = PointCloud<PointXYZ>::Ptr(new PointCloud<PointXYZ>());
	PointCloud<Normal>::Ptr	normal = PointCloud<Normal>::Ptr(new PointCloud<Normal>());
	io::cloud_load_point_cloud(basepath + filename, basepath, cloud, normal);

	PointCloud<PointXYZ>::Ptr out_cloud = PointCloud<PointXYZ>::Ptr(new PointCloud<PointXYZ>());
	UniformOctreeSampling<PointXYZ> sampler;
	sampler.setSamplingResolution(5);
	sampler.setSampleRadiusSearch(5);
	sampler.setOctreeResolution(5);
	sampler.setInputCloud(cloud);
	sampler.setInputNormalCloud(normal);
	sampler.setOctreeNormalThreshold(0.8);
	sampler.filter(*out_cloud);
}

vtkSmartPointer<vtkActor> vtk_build_box_actor(std::vector<std::pair<Vector3f, Vector3f>> &bounds, Vector3f color = Vector3f(1.0f, 0.0f, 0.0f), float linewidth = 1.0f)
{
	static const float degenerate_threshold = 0.5f;
	static const float box_offsets[8][3] = { { 0, 0, 0 }, { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 }, { 1, 1, 0 }, { 0, 1, 1 }, { 1, 0, 1 }, { 1, 1, 1 } };
	static const int   box_edges[12][2] = {{ 0, 1 }, {0, 2 }, { 0, 3 }, { 4, 7 }, { 5, 7 }, { 6, 7 }, { 1, 6 }, { 2, 5 }, { 2, 4 }, { 3, 6 }, { 3, 5 }, {1, 4}};

	size_t nbox = bounds.size();
	auto vtk_pcoords = vtkSmartPointer<vtkFloatArray>::New();
	vtk_pcoords->SetNumberOfComponents(3);
	vtk_pcoords->SetNumberOfTuples(nbox * 8);
	for(size_t i = 0; i < nbox; ++i)
	{
		Vector3f bmin = bounds[i].first;
		Vector3f bmax = bounds[i].second;
		Vector3f delta = bmax - bmin;
		for(size_t j = 0; j <3; ++j)
			delta[j] = std::max(delta[j], degenerate_threshold);

		for (size_t j = 0; j < 8; ++j)
		{
			float x = bmin[0] + box_offsets[j][0] * delta[0];
			float y = bmin[1] + box_offsets[j][1] * delta[1];
			float z = bmin[2] + box_offsets[j][2] * delta[2];
			vtk_pcoords->SetTuple3(i * 8 + j, x, y, z);
		}
	}

	auto vtk_points = vtkSmartPointer<vtkPoints>::New();
	vtk_points->SetData(vtk_pcoords);

	auto lines = vtkSmartPointer<vtkCellArray>::New();
	lines->Allocate(nbox * 12 * 2 + nbox * 12, 1000);
	for (size_t i = 0; i < nbox; ++i)
	{
		for (size_t e = 0; e < 12; ++e)
		{
			lines->InsertNextCell(2);
			lines->InsertCellPoint(i * 8 + box_edges[e][0]);
			lines->InsertCellPoint(i * 8 + box_edges[e][1]);
		}
	}

	vtkSmartPointer<vtkPolyData> polydata = vtkSmartPointer<vtkPolyData>::New();
	polydata->SetPoints(vtk_points);
	polydata->SetLines(lines);

	vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
	mapper->SetInputData(polydata);

	auto actor = vtkSmartPointer<vtkActor>::New();
	actor->SetMapper(mapper);
	actor->GetProperty()->SetColor(color(0), color(1), color(2));
	actor->GetProperty()->SetLineWidth(linewidth);
	
	return actor;
}

vtkSmartPointer<vtkActor> vtk_build_points_actor(std::vector<Vector3f> &points, Vector3f color = Vector3f(1.0f, 0.0f, 0.0f), float size = 1.0f)
{
	vtkSmartPointer<vtkPoints> point_arr = vtkSmartPointer<vtkPoints>::New();
	point_arr->SetNumberOfPoints(points.size());
	for (size_t i = 0; i < points.size(); ++i)
	{
		point_arr->SetPoint(i, points[i].data());
	}

	vtkSmartPointer<vtkCellArray> vertices = vtkSmartPointer<vtkCellArray>::New();
	for (size_t i = 0; i < points.size(); ++i)
	{
		vertices->InsertNextCell(1);
		vertices->InsertCellPoint(i);
	}

	vtkSmartPointer<vtkPolyData> polydata = vtkSmartPointer<vtkPolyData>::New();
	polydata->SetPoints(point_arr);
	polydata->SetVerts(vertices);

	vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
	mapper->SetInputData(polydata);

	auto actor = vtkSmartPointer<vtkActor>::New();
	actor->SetMapper(mapper);
	actor->GetProperty()->SetColor(color(0), color(1), color(2));
	actor->GetProperty()->SetPointSize(size);

	return actor;
}

vtkSmartPointer<vtkActor> pcl_build_point_cloud_actor(PointCloud<PointXYZ>::Ptr cloud)
{
	std::vector<Vector3f> points(cloud->size());
	for (size_t i = 0; i < cloud->size(); ++i)
	{
		points[i] = cloud->points[i].getVector3fMap();
	}

	return vtk_build_points_actor(points, Vector3f(1.0f, 1.0f, 1.0f), 2.0f);
}

int main()
{


	string filename = "normal_lucy_none-Slice-54_center_vn.obj";
	string basepath = "G:\\Projects\\Oh\\data\\test_data\\";
	PointCloud<PointXYZ>::Ptr cloud = PointCloud<PointXYZ>::Ptr(new PointCloud<PointXYZ>());
	PointCloud<Normal>::Ptr	normal = PointCloud<Normal>::Ptr(new PointCloud<Normal>());
	io::cloud_load_point_cloud(basepath + filename, basepath, cloud, normal);

	PointCloud<PointXYZ>::Ptr out_cloud = PointCloud<PointXYZ>::Ptr(new PointCloud<PointXYZ>());
	UniformOctreeSampling<PointXYZ> sampler;
	sampler.setSamplingResolution(5);
	sampler.setSampleRadiusSearch(5);
	sampler.setOctreeResolution(5);
	sampler.setInputCloud(cloud);
	sampler.setInputNormalCloud(normal);
	sampler.setOctreeNormalThreshold(0.8);
	sampler.filter(*out_cloud);

	auto sample_actor = vtk_build_points_actor(sampler.test_sample_points, Vector3f(1.0f, 0.0f, 0.0f), 2.0f);
	auto sample_1_actor = vtk_build_points_actor(sampler.test_sample_points_1, Vector3f(0.0f, 1.0f, 1.0f), 2.0f);
	auto node_point_actor = vtk_build_points_actor(sampler.test_node_points, Vector3f(1.0f, 1.0f, 0.0f), 2.0f);
	auto node_bb_actor = vtk_build_box_actor(sampler.test_node_bounds, Vector3f(0.3f, .6f, 0.1f));
	auto cloud_actor = pcl_build_point_cloud_actor(cloud);


	vtkRenderer *ren1 = vtkRenderer::New();
	ren1->SetBackground(0.4, 0.4, 0.4);

	ren1->AddActor(sample_actor);
	ren1->AddActor(sample_1_actor);
	ren1->AddActor(cloud_actor);
	ren1->AddActor(node_point_actor);
	ren1->AddActor(node_bb_actor);

	vtkRenderWindow *renWin = vtkRenderWindow::New();
	renWin->AddRenderer(ren1);
	renWin->SetSize(800, 600);
		
	vtkRenderWindowInteractor *iren = vtkRenderWindowInteractor::New();
	iren->SetRenderWindow(renWin);

	vtkInteractorStyleTrackballCamera *style =
		vtkInteractorStyleTrackballCamera::New();
	iren->SetInteractorStyle(style);

	iren->Initialize();
	iren->Start();

	return 0;
}