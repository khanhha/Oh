///***********************************************************************
//* Software License Agreement (BSD License)
//*
//* Copyright 2011-2016 Jose Luis Blanco (joseluisblancoc@gmail.com).
//*   All rights reserved.
//*
//* Redistribution and use in source and binary forms, with or without
//* modification, are permitted provided that the following conditions
//* are met:
//*
//* 1. Redistributions of source code must retain the above copyright
//*    notice, this list of conditions and the following disclaimer.
//* 2. Redistributions in binary form must reproduce the above copyright
//*    notice, this list of conditions and the following disclaimer in the
//*    documentation and/or other materials provided with the distribution.
//*
//* THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
//* IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
//* OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
//* IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
//* INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
//* NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
//* DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
//* THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
//* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
//* THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//*************************************************************************/
//
//#include <pcl/kdtree/nanoflann.hpp>
//#include <pcl/kdtree/kdtree_flann.h>
//#include "io/tiny_obj_loader.h"
//
//#include <ctime>
//#include <cstdlib>
//#include <iostream>
//
//using namespace std;
//using namespace nanoflann;
//using namespace pcl;
//
//void dump_mem_usage();
//
//// This is an exampleof a custom data set class
//template <typename T>
//struct PointCloud_T
//{
//	typedef T coord_t; //!< The type of each coordinate
//
//	struct Point
//	{
//		T  x, y, z;
//	};
//	struct Normal
//	{
//		T x, y, z;
//	};
//
//	std::vector<Point>  pts;
//	std::vector<Normal> nms;
//}; // end of PointCloud
//
//   // And this is the "dataset to kd-tree" adaptor class:
//template <typename Derived>
//struct PointCloudAdaptor
//{
//	typedef typename Derived::coord_t coord_t;
//
//	const Derived &obj; //!< A const ref to the data set origin
//
//						/// The constructor that sets the data set source
//	PointCloudAdaptor(const Derived &obj_) : obj(obj_) { }
//
//	/// CRTP helper method
//	inline const Derived& derived() const { return obj; }
//
//	// Must return the number of data points
//	inline size_t kdtree_get_point_count() const { return derived().pts.size(); }
//
//	// Returns the dim'th component of the idx'th point in the class:
//	// Since this is inlined and the "dim" argument is typically an immediate value, the
//	//  "if/else's" are actually solved at compile time.
//	inline coord_t kdtree_get_pt(const size_t idx, int dim) const
//	{
//		if (dim == 0) return derived().pts[idx].x;
//		else if (dim == 1) return derived().pts[idx].y;
//		else return derived().pts[idx].z;
//	}
//
//	inline void kdtree_get_nm(const size_t idx, coord_t &x, coord_t &y, coord_t &z) const
//	{
//		x = derived().nms[idx].x;
//		y = derived().nms[idx].y;
//		z = derived().nms[idx].z;
//	}
//
//	// Optional bounding-box computation: return false to default to a standard bbox computation loop.
//	//   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it again.
//	//   Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
//	template <class BBOX>
//	bool kdtree_get_bbox(BBOX& /*bb*/) const { return false; }
//
//}; // end of PointCloudAdaptor
//
//
//template <typename T>
//void generateRandomPointCloud(PointCloud_T<T> &point, const size_t N, const T max_range = 10)
//{
//	std::cout << "Generating " << N << " point cloud...";
//	point.pts.resize(N);
//	for (size_t i = 0; i < N; i++)
//	{
//		point.pts[i].x = max_range * (rand() % 1000) / T(1000);
//		point.pts[i].y = max_range * (rand() % 1000) / T(1000);
//		point.pts[i].z = max_range * (rand() % 1000) / T(1000);
//	}
//
//	std::cout << "done\n";
//}
//
//template <typename T>
//void load_pointcloud(const string &filename, const string &basepath, PointCloud_T<T> &cloud)
//{
//	tinyobj::attrib_t attrib;
//	std::vector<tinyobj::shape_t> shapes;
//	std::vector<tinyobj::material_t> materials;
//	std::string err;
//
//	bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &err, filename.c_str(), basepath.c_str(), false);
//
//	if (ret)
//	{
//		for (size_t v = 0; v < attrib.vertices.size() / 3; v++) {
//			PointCloud_T<T>::Point  p{ attrib.vertices[3 * v + 0], attrib.vertices[3 * v + 1], attrib.vertices[3 * v + 2] };
//			PointCloud_T<T>::Normal n{ attrib.normals[3 * v + 0], attrib.normals[3 * v + 1], attrib.normals[3 * v + 2] };
//			cloud.pts.push_back(p);
//			cloud.nms.push_back(n);
//		}
//	}
//}
//
//template <typename num_t>
//void kdtree_demo(const size_t N)
//{
//	PointCloud_T<num_t> cloud;
//	string filename = "normal_lucy_none-Slice-54_center_vn.obj";
//	string basepath = "G:\\Projects\\Oh\\data\\test_data\\";
//
//	// Generate points:
//	//generateRandomPointCloud_T(cloud, N);
//	load_pointcloud(basepath + filename, basepath, cloud);
//
//	num_t query_pt[3] = { -36, 79, 1321};
//
//	typedef PointCloudAdaptor<PointCloud_T<num_t> > PC2KD;
//	const PC2KD  pc2kd(cloud); // The adaptor
//
//							   // construct a kd-tree index:
//	typedef KDTreeSingleIndexNormalAdaptor<
//		L2_Simple_Adaptor<num_t, PC2KD >,
//		PC2KD,
//		3 /* dim */
//	> my_kd_tree_t;
//
//	dump_mem_usage();
//
//	my_kd_tree_t   index(3 /*dim*/, pc2kd, KDTreeSingleIndexNormalAdaptorParams(30 /* max leaf */, 0.9f));
//	index.buildIndex();
//	dump_mem_usage();
//
//	// do a knn search
//	const size_t num_results = 1;
//	size_t ret_index;
//	num_t out_dist_sqr;
//	nanoflann::KNNResultSet<num_t> resultSet(num_results);
//	resultSet.init(&ret_index, &out_dist_sqr);
//	index.findNeighbors(resultSet, &query_pt[0], nanoflann::SearchParams(10));
//	
//	std::vector<num_t> dists(10);
//	std::vector<size_t> indices(10);
//	index.knnSearch(&query_pt[0], 10, &indices[0], &dists[0], num_results);
//
//	std::cout << "knnSearch(nn=" << num_results << "): \n";
//	std::cout << "ret_index=" << ret_index << " out_dist_sqr=" << out_dist_sqr << endl;
//
//}
//
//int main__()
//{
//	// Randomize Seed
//	srand(time(NULL));
//	kdtree_demo<float>(1000000);
//	kdtree_demo<double>(1000000);
//
//
//	string filename = "normal_lucy_none-Slice-54_center_vn.obj";
//	string basepath = "G:\\Projects\\Oh\\data\\test_data\\";
//	// Generate points:
//	//generateRandomPointCloud(cloud, N);
//	PointCloud<PointXYZ>::Ptr cloud = PointCloud<PointXYZ>::Ptr(new PointCloud<PointXYZ>());
//	//load_pointcloud(basepath + filename, basepath, cloud);
//
//	char pause;
//	std::cin >> pause; 
//	return 0;
//}
//
//void dump_mem_usage()
//{
//	FILE* f = fopen("/proc/self/statm", "rt");
//	if (!f) return;
//	char str[300];
//	size_t n = fread(str, 1, 200, f);
//	str[n] = 0;
//	printf("MEM: %s\n", str);
//	fclose(f);
//}
