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
	float max_scl;
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


string enumToString(OctreeSampling<PointXYZ>::ResampleMethod method)
{
	switch (method)
	{
	case pcl::OctreeSampling<pcl::PointXYZ>::ResampleMethod::UNIFORM:
		return "uniform";
		break;
	case pcl::OctreeSampling<pcl::PointXYZ>::ResampleMethod::NONUNIFORM_MAX_POINTS_PER_LEAF:
		return "nonuniform_max_point_per_leaf";
		break;
	case pcl::OctreeSampling<pcl::PointXYZ>::ResampleMethod::NONUNIFORM_NORMAL_THRESHOLD:
		return "nonuniform_normal_threshodl";
		break;
	default:
		return "";
		break;
	}
}
string enumToString(OctreeSampling<PointXYZ>::InterpolationMethod method)
{
	switch (method)
	{
	case pcl::OctreeSampling<pcl::PointXYZ>::InterpolationMethod::CLOSEST_TO_CENTER:
		return "closest_to_center";
		break;
	case pcl::OctreeSampling<pcl::PointXYZ>::InterpolationMethod::AVERAGE:
		return "average";
		break;
	case pcl::OctreeSampling<pcl::PointXYZ>::InterpolationMethod::HEIGHT_INTERPOLATION:
		return "height_interpolation";
		break;
	default:
		return "";
		break;
	}
}

vtkRenderer *g_ren1 = nullptr;
void test_octree_resampling()
{
	string filenames[] = {
		"normal_lucy_none-Slice-54_center_vn"
		//"normal_lucy_none-Slice-55_center_vn",
		//"normal_lucy_none-Slice-56_center_vn",
		//"normal_lucy_none-Slice-57_center_vn",
		//"normal_lucy_tshirt-Slice-54_center_vn",
		//"normal_lucy_tshirt-Slice-55_center_vn",
		//"normal_lucy_tshirt-Slice-56_center_vn",
		//"normal_lucy_tshirt-Slice-57_center_vn",
		//"normal_lucy_none_repaired",
		//"normal_lucy_standard_tee_repaired"
	};


	size_t nfiles = sizeof(filenames)/sizeof(string);
	for (size_t i = 0; i < nfiles; ++i)
	{
		string filename = filenames[i];
		string basepath = "G:\\Projects\\Oh\\data\\test_data\\";
		PointCloud<PointXYZ>::Ptr cloud = PointCloud<PointXYZ>::Ptr(new PointCloud<PointXYZ>());
		PointCloud<Normal>::Ptr	normal = PointCloud<Normal>::Ptr(new PointCloud<Normal>());
		io::cloud_load_point_cloud(basepath + filename + ".obj", basepath, cloud, normal);

		OctreeSampling<PointXYZ>::ResampleMethod resample_med = OctreeSampling<PointXYZ>::ResampleMethod::UNIFORM;
		OctreeSampling<PointXYZ>::InterpolationMethod inter_med = OctreeSampling<PointXYZ>::InterpolationMethod::HEIGHT_INTERPOLATION;

		OctreeSampling<PointXYZ> sampler;
		PointCloud<PointXYZ>::Ptr out_cloud = PointCloud<PointXYZ>::Ptr(new PointCloud<PointXYZ>());
		sampler.setResampleMethod(resample_med);
		sampler.setInterpolationMethod(inter_med);
		//sampler.setMaxPointsPerLeaf(6);
		sampler.setSamplingResolution(5);
		//sampler.setSampleRadiusSearch(5);
		//sampler.setOctreeResolution(0.005);
		//sampler.setOctreeNormalThreshold(0.9);

		sampler.setInputCloud(cloud);
		sampler.setInputNormalCloud(normal);
		sampler.filter(*out_cloud);

		write_obj_points("G:\\Projects\\Oh\\data\\resample_result\\" + filename + "_" + enumToString(resample_med) + "_"+ enumToString(inter_med) + ".obj", *out_cloud);
	
#if 1
		auto sample_actor = vtk_build_points_actor(out_cloud->points, Vector3f(1.0f, 0.0f, 0.0f), 6.0f);
		//auto sample_1_actor = vtk_build_points_actor(sampler.test_sample_points_1, Vector3f(0.0f, 1.0f, 1.0f), 6.0f);
		//auto sample_2_actor = vtk_build_points_actor(sampler.test_sample_points_2, Vector3f(0.0f, 0.0f, 1.0f), 6.0f);
		//auto node_point_actor = vtk_build_points_actor(sampler.test_node_points, Vector3f(1.0f, 1.0f, 0.0f), 3.0f);
		//auto node_bb_actor = vtk_build_box_actor(sampler.test_node_bounds, Vector3f(0.3f, .6f, 0.1f));
		//auto cloud_actor = pcl_build_point_cloud_actor(cloud);
		//auto node_text_actors = vtk_build_number_text(sampler.test_node_ids, 0.02);

		g_ren1->SetBackground(0.4, 0.4, 0.4);

		g_ren1->AddActor(sample_actor);
		//g_ren1->AddActor(sample_1_actor);
		//g_ren1->AddActor(sample_2_actor);
		//g_ren1->AddActor(cloud_actor);
		//g_ren1->AddActor(node_point_actor);
		//g_ren1->AddActor(node_bb_actor);
		//for (auto ac : node_text_actors)
		//	g_ren1->AddViewProp(ac);
#endif
	}

	//string filename = "Armadillo"; //1 resolution
	//string filename = "normal_oh_none_repaired_points"; //4 resolution
	//string filename = "lucy_none-Slice-54_center_vn";


}

void test_weight_sampling()
{
#if 0
	string filenames[] = {
		"normal_lucy_none-Slice-54_center_vn",
		"normal_lucy_none-Slice-55_center_vn",
		"normal_lucy_none-Slice-56_center_vn",
		"normal_lucy_none-Slice-57_center_vn",
		"normal_lucy_tshirt-Slice-54_center_vn",
		"normal_lucy_tshirt-Slice-55_center_vn",
		"normal_lucy_tshirt-Slice-56_center_vn",
		"normal_lucy_tshirt-Slice-57_center_vn",
		"normal_lucy_none_repaired",
		"normal_lucy_standard_tee_repaired"
	};

	size_t nfiles = sizeof(filenames) / sizeof(string);
	for (size_t i = 0; i < nfiles; ++i)
	{
		string filename = filenames[i];
		string basepath = "D:\\Projects\\Oh\\data\\test_data\\";
		PointCloud<PointXYZ>::Ptr cloud = PointCloud<PointXYZ>::Ptr(new PointCloud<PointXYZ>());
		PointCloud<Normal>::Ptr	normal = PointCloud<Normal>::Ptr(new PointCloud<Normal>());
		io::cloud_load_point_cloud(basepath + filename + ".obj", basepath, cloud, normal);
		
		PointCloud<PointXYZ>::Ptr out_cloud = PointCloud<PointXYZ>::Ptr(new PointCloud<PointXYZ>());
		WeightSampling<PointXYZ> sampler;
		sampler.setInputCloud(cloud);
		sampler.setKNeighbourSearch(10);
		sampler.setSigma(1.5);
		sampler.setResamplePercent(0.5);
		sampler.filter(*out_cloud);

		write_obj_points("D:\\Projects\\Oh\\data\\resample_weight_result\\" + filename +  "_graph_resampled.obj", *out_cloud);
	}
#else
	string filename = "normal_lucy_standard_tee_repaired";
	//string filename = "cube";
	//string filename = "Armadillo_points";
	//string filename = "normal_oh_none_repaired";
	//string filename = "lucy_none-Slice-54_center_vn";
	string basepath = "D:\\Projects\\Oh\\data\\test_data\\";
	PointCloud<PointXYZ>::Ptr cloud = PointCloud<PointXYZ>::Ptr(new PointCloud<PointXYZ>());
	PointCloud<Normal>::Ptr	normal = PointCloud<Normal>::Ptr(new PointCloud<Normal>());
	io::cloud_load_point_cloud(basepath + filename+".obj", basepath, cloud, normal);

	PointCloud<PointXYZ>::Ptr out_cloud = PointCloud<PointXYZ>::Ptr(new PointCloud<PointXYZ>());
	WeightSampling<PointXYZ> sampler;
	sampler.setInputCloud(cloud);
	//sampler.setKNeighbourSearch(10);
	sampler.setRadiusSearch(5);
	sampler.setSigma(1.5);
	sampler.setResamplePercent(0.2);
	sampler.filter(*out_cloud);

	//std::vector<Vector3i> colors = scalarToColor(sampler.test_weights);
	write_obj_points("D:\\Projects\\Oh\\data\\resample_weight_result\\" + filename + "_graph_resampled.obj", *out_cloud);

	auto cloud_actor = vtk_build_points_actor(cloud->points, Vector3f(1.0f, 1.0f, 1.0f), 1.0f);
	auto sample_actor = vtk_build_points_actor(out_cloud->points, Vector3f(1.0f, 1.0f, 1.0f), 1.0f);
	auto test_point_actor = vtk_build_points_actor(sampler.test_sample_points, Vector3f(0.0f, 1.0f, 1.0f), 5.0f);
	
	auto segments_actor = vtk_build_segments_actor(sampler.test_segments);
	g_ren1->AddActor(cloud_actor);
	g_ren1->AddActor(sample_actor);
	g_ren1->AddActor(segments_actor);
	//g_ren1->AddActor(test_point_actor);
#if 0
	std::vector<std::pair<Vector3f, int>> text_points;
	for (size_t i = 0; i < cloud->points.size(); i++)
	{
		if (i >= 111000 && i < 116000)
		{
			if (i == 113471 || i == 113472)
				text_points.push_back(std::make_pair(cloud->points[i].getVector3fMap(), i));
		}
	}
	auto point_id_text_actor = vtk_build_number_text(text_points, 0.1);
	for (auto ac : point_id_text_actor)
		g_ren1->AddActor(ac);
#endif

#endif
}

void test_uniform_sampling()
{
	string filename = "Armadillo.obj";
	//string filename = "normal_oh_none_repaired.obj";
	//string filename = "lucy_none-Slice-54_center_vn.obj";
	string basepath = "G:\\Projects\\Oh\\data\\test_data\\";
	PointCloud<PointXYZ>::Ptr cloud = PointCloud<PointXYZ>::Ptr(new PointCloud<PointXYZ>());
	PointCloud<Normal>::Ptr	normal = PointCloud<Normal>::Ptr(new PointCloud<Normal>());
	io::cloud_load_point_cloud(basepath + filename, basepath, cloud, normal);

	PointCloud<PointXYZ>::Ptr out_cloud = PointCloud<PointXYZ>::Ptr(new PointCloud<PointXYZ>());
	UniformSampling<PointXYZ> sampler;
	sampler.setInputCloud(cloud);
	sampler.setRadiusSearch(1.5);
	sampler.filter(*out_cloud);


	std::vector<Vector3f> sampler_points;
	for (size_t i = 0; i < out_cloud->size(); ++i)
	{
		sampler_points.push_back(out_cloud->points[i].getArray3fMap());
	}

	auto cloud_actor = pcl_build_point_cloud_actor(cloud);
	auto sample_actor = vtk_build_points_actor(sampler_points, Vector3f(1.0f, 0.0f, 0.0f), 3.0f);
	g_ren1->AddActor(cloud_actor);
	g_ren1->AddActor(sample_actor);
}

int main()
{
	g_ren1 = vtkRenderer::New();
	g_ren1->SetBackground(0.4, 0.4, 0.4);

	test_weight_sampling();
	//test_uniform_sampling();
	//test_octree_resampling();

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