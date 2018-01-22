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
VTK_MODULE_INIT(vtkRenderingFreeType);
VTK_MODULE_INIT(vtkInteractionStyle);

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
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkTextActor.h>
#include <vtkTextProperty.h>
#include <vtkTextActor3D.h>

// For compatibility with new VTK generic data arrays
#ifdef vtkGenericDataArray_h
#define InsertNextTupleValue InsertNextTypedTuple
#endif

#include <string>
#include <algorithm>
#include <pcl/point_cloud.h>
#include <pcl/io/obj_loader.h>
#include <pcl/filters/uniform_sampling.h>
#include <pcl/filters/octree_sampling.h>
#include <pcl/filters/weight_sampling.h>
#include <pcl/surface/convex_hull.h>

#include <Eigen/Dense>

using namespace  std;
using namespace  Eigen;
using namespace  pcl;

typedef vtkSmartPointer<vtkActor> ActorPtr;
typedef vtkSmartPointer<vtkTextActor3D> Actor3DPtr;

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

void write_obj_points(std::string filename, const std::vector<Vector3f> &points)
{
	std::ofstream of(filename);
	for (auto &p : points)
	{
		of << "v " << p[0] << " " << p[1] << " " << p[2] << std::endl;
	}
	of.close();
}

void write_obj_points(std::string filename, const PointCloud<PointXYZ> &cloud, const std::vector<Vector3i> &colors)
{
	std::ofstream of(filename);
	for (size_t i = 0; i < cloud.points.size(); ++i)
	{
		auto &p = cloud.points[i];
		Vector3f c = colors[i].cast<float>();
		c = c / 255.0f;
		of << "v " << p.x << " " << p.y << " " << p.z << " " << c[0] << " " << c[1] << " " << c[2] << std::endl;
	}
	of.close();
}

void write_obj_points(std::string filename, const PointCloud<PointXYZ> &cloud, Vector3f color = Vector3f(1.0f, 0.0f, 0.0f))
{
	std::ofstream of(filename);
	for (auto &p : cloud.points)
	{
		of << "v " << p.x << " " << p.y << " " << p.z << " " <<color[0] <<" " << color[1] <<" " << color[2] <<std::endl;
	}
	of.close();
}

void test_uniform_octree_sample()
{
	string filename = "normal_lucy_none-Slice-54_center_vn.obj";
	string basepath = "G:\\Projects\\Oh\\data\\test_data\\";
	PointCloud<PointXYZ>::Ptr cloud = PointCloud<PointXYZ>::Ptr(new PointCloud<PointXYZ>());
	PointCloud<Normal>::Ptr	normal = PointCloud<Normal>::Ptr(new PointCloud<Normal>());
	io::cloud_load_point_cloud(basepath + filename, basepath, cloud, normal);

	PointCloud<PointXYZ>::Ptr out_cloud = PointCloud<PointXYZ>::Ptr(new PointCloud<PointXYZ>());
	OctreeSampling<PointXYZ> sampler;
	sampler.setSamplingResolution(5);
	sampler.setSampleRadiusSearch(5);
	sampler.setOctreeResolution(5);
	sampler.setInputCloud(cloud);
	sampler.setInputNormalCloud(normal);
	sampler.setOctreeNormalThreshold(0.8);
	sampler.filter(*out_cloud);
}

vtkSmartPointer<vtkActor> vtk_build_segments_actor(std::vector<Vector3f> &segments, Vector3f color = Vector3f(1.0f, 0.0f, 0.0f), float linewidth = 3.0f)
{
	size_t npoints = segments.size();
	size_t nsegments = npoints / 2;
	auto vtk_pcoords = vtkSmartPointer<vtkFloatArray>::New();
	vtk_pcoords->SetNumberOfComponents(3);
	vtk_pcoords->SetNumberOfTuples(npoints);
	for (size_t i = 0; i < npoints; ++i)
	{
		auto &p = segments[i];
		vtk_pcoords->SetTuple3(i, p[0], p[1], p[2]);
	}

	auto vtk_points = vtkSmartPointer<vtkPoints>::New();
	vtk_points->SetData(vtk_pcoords);

	auto lines = vtkSmartPointer<vtkCellArray>::New();
	lines->Allocate(npoints, 1000);
	for (size_t i = 0; i < nsegments; ++i)
	{
		lines->InsertNextCell(2);
		lines->InsertCellPoint(i * 2);
		lines->InsertCellPoint(i * 2 + 1);
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


vtkSmartPointer<vtkActor> vtk_build_points_actor(std::vector<Vector3f> &points, Vector3f color = Vector3f(1.0f, 0.0f, 0.0f), float size = 1.0f, const std::vector<Vector3i> &vert_colors = std::vector<Vector3i>())
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
	
	auto mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
	mapper->SetInputData(polydata);
	auto actor = vtkSmartPointer<vtkActor>::New();
	
	if (!vert_colors.empty())
	{
		vtkSmartPointer<vtkUnsignedCharArray> colors = vtkSmartPointer<vtkUnsignedCharArray>::New();
		colors->SetNumberOfComponents(3);
		colors->SetName("Colors");
		for (size_t i = 0; i < points.size(); ++i)
		{
			unsigned char c[3] = { vert_colors[i][0], vert_colors[i][1], vert_colors[i][2] };
			colors->InsertNextTupleValue(c);
		}

		polydata->GetPointData()->SetScalars(colors);
	}
	else {
		actor->GetProperty()->SetColor(color(0), color(1), color(2));
	}

	actor->SetMapper(mapper);
	actor->GetProperty()->SetPointSize(size);

	return actor;
}


vtkSmartPointer<vtkActor> vtk_build_points_actor(const std::vector<PointXYZ, Eigen::aligned_allocator<PointXYZ> > &points, Vector3f color = Vector3f(1.0f, 0.0f, 0.0f), float size = 1.0f)
{
	std::vector<Vector3f> e_points;
	for (const auto &p : points)
	{
		Vector3f tmp(p.x, p.y, p.z);
		e_points.push_back(tmp);
	}

	return vtk_build_points_actor(e_points, color, size);
}

std::vector<Vector3i> scalarToColor(const std::vector<float> &color_scalars)
{
	std::vector<Vector3i> color_points;
	std::vector<float> sorted_scalars = color_scalars;
	std::sort(sorted_scalars.begin(), sorted_scalars.end());
	float rangemin = sorted_scalars[0.1 * sorted_scalars.size()];
	float rangemax = sorted_scalars[0.9 * sorted_scalars.size()];
	//if (!color_scalars.empty())
	//	max_scl = *(std::max_element(color_scalars.begin(), color_scalars.end()));

	for (auto i = 0; i < color_scalars.size(); ++i)
	{
		Vector3i c;
		if (!color_scalars.empty()) 
		{
			float val = color_scalars[i];
			//val = std::clamp(val, rangemin, rangemax);
			val = std::max(val, rangemin);
			val = std::min(val, rangemax);
			float scale = (val - rangemin) / (rangemax - rangemin);
			c = Vector3i(255 * scale, 0, 0);
		}
		else
			c = Vector3i(255, 0, 0);

		color_points.push_back(c);
	}

	return color_points;
}

vtkSmartPointer<vtkActor> vtk_build_points_actor(const std::vector<PointXYZ, Eigen::aligned_allocator<PointXYZ> > &points, 
	std::vector<float> color_scalars, float size = 1.0f)
{
	std::vector<Vector3f> e_points;
	std::vector<Vector3i> color_points = scalarToColor(color_scalars);

	Vector3f color = Vector3f(1.0f, 0, 0);
	for (auto i =  0; i < points.size(); ++i)
	{
		auto &p = points[i];
		Vector3f tmp(p.x, p.y, p.z);
		e_points.push_back(tmp);
	}
	return vtk_build_points_actor(e_points, color, size, color_points);
}

std::vector<Actor3DPtr> vtk_build_number_text(std::vector<std::pair<Eigen::Vector3f, int>> ids, float scale = 0.01)
{
	std::vector<Actor3DPtr> texts;
	for (int i = 0; i < ids.size(); ++i)
	{
		Actor3DPtr actor = Actor3DPtr::New();
		stringstream ss; ss << ids[i].second;
		string str = ss.str();
		actor->SetInput(str.c_str());
		Eigen::Vector3d p = ids[i].first.cast<double>();
		actor->SetPosition(p.data());
		actor->SetScale(scale);
		
		texts.push_back(actor);
	}
	return texts;
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

vtkRenderer *g_ren1 = nullptr;

void test_convex_hull()
{
	//string filename = "Armadillo.obj";
	//string filename = "normal_oh_none_repaired.obj";
	string filename = "lucy_none-Slice-54_center_vn.obj";
	string basepath = "D:\\Projects\\Oh\\data\\test_data\\";
	PointCloud<PointXYZ>::Ptr cloud = PointCloud<PointXYZ>::Ptr(new PointCloud<PointXYZ>());
	PointCloud<Normal>::Ptr	normal = PointCloud<Normal>::Ptr(new PointCloud<Normal>());
	io::cloud_load_point_cloud(basepath + filename, basepath, cloud, normal);


	PointCloud<PointXYZ>::Ptr  out_hull = PointCloud<PointXYZ>::Ptr(new PointCloud<PointXYZ>());
	pcl::ConvexHull<PointXYZ> chull_builder;
	chull_builder.setInputCloud(cloud);
	chull_builder.setDimension(2);
	chull_builder.setComputeAreaVolume(true);
	chull_builder.reconstruct(*out_hull);


	auto cloud_actor = vtk_build_points_actor(cloud->points, Vector3f(1.0f, 1.0f, 1.0f), 1.0f);
	auto sample_actor = vtk_build_points_actor(out_hull->points, Vector3f(1.0f, 0.0f, 0.0f), 4.0f);
	g_ren1->AddActor(cloud_actor);
	g_ren1->AddActor(sample_actor);
}

int main()
{
	g_ren1 = vtkRenderer::New();
	g_ren1->SetBackground(0.4, 0.4, 0.4);

	test_convex_hull();

	vtkRenderWindow *renWin = vtkRenderWindow::New();
	renWin->AddRenderer(g_ren1);
	renWin->SetSize(800, 600);
		
	vtkRenderWindowInteractor *iren = vtkRenderWindowInteractor::New();
	iren->SetRenderWindow(renWin);

	vtkInteractorStyleTrackballCamera *style = vtkInteractorStyleTrackballCamera::New();
	iren->SetInteractorStyle(style);

	iren->Initialize();
	iren->Start();

	return 0;
}