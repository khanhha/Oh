/***********************************************************************
* Software License Agreement (BSD License)
*
* Copyright 2011-2016 Jose Luis Blanco (joseluisblancoc@gmail.com).
*   All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions
* are met:
*
* 1. Redistributions of source code must retain the above copyright
*    notice, this list of conditions and the following disclaimer.
* 2. Redistributions in binary form must reproduce the above copyright
*    notice, this list of conditions and the following disclaimer in the
*    documentation and/or other materials provided with the distribution.
*
* THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
* IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
* OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
* IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
* INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
* NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
* DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
* THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
* THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*************************************************************************/

#include <pcl/kdtree/nanoflann.hpp>
#include <pcl/kdtree/kdtree_flann.h>
#include "io/tiny_obj_loader.h"

#include <ctime>
#include <cstdlib>
#include <iostream>

using namespace std;
using namespace nanoflann;
using namespace pcl;

void load_pcl_1(const string &filename, const string &basepath, PointCloud<PointXYZ>::Ptr cloud, PointCloud<Normal>::Ptr normal)
{
	tinyobj::attrib_t attrib;
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;
	std::string err;

	bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &err, filename.c_str(), basepath.c_str(), false);

	if (ret)
	{
		for (size_t v = 0; v < attrib.vertices.size() / 3; v++) {
			PointXYZ p(attrib.vertices[3 * v + 0], attrib.vertices[3 * v + 1], attrib.vertices[3 * v + 2]);
			cloud->push_back(p);
			Normal n(attrib.normals[3 * v + 0], attrib.normals[3 * v + 1], attrib.normals[3 * v + 2]);
			normal->push_back(n);
		}
	}
}

int main()
{
	// Randomize Seed
	srand(time(NULL));

	string filename = "normal_lucy_none-Slice-54_center_vn.obj";
	string basepath = "G:\\Projects\\Oh\\data\\test_data\\";

	PointCloud<PointXYZ>::Ptr cloud = PointCloud<PointXYZ>::Ptr(new PointCloud<PointXYZ>());
	PointCloud<Normal>::Ptr	normal = PointCloud<Normal>::Ptr(new PointCloud<Normal>());
	load_pcl_1(basepath + filename, basepath, cloud, normal);

	KdTreeFLANN<PointXYZ> kdtree;
	kdtree.setInputCloud(cloud);
	kdtree.setNormalCloud(normal);
	kdtree.setMaxPointsPerLeaf(30);
	kdtree.setNormalThreshold(0.8f);
	kdtree.addPointsFromInputCloud();

	PointXYZ p = cloud->points[10];
	std::vector<int> indices;
	std::vector<float> sqrdsts;
	kdtree.nearestKSearch(p, 10, indices, sqrdsts);

	std::vector<int> indices_1;
	std::vector<float> sqrdist_1;
	kdtree.radiusSearch(p, 10.0f, indices_1, sqrdist_1);

	std::vector<float> bmin, bmax;
	size_t nleaf = kdtree.getAllLeafNodesBoundingBox(bmin, bmax);

	kdtree.getNodesBoundingBoxAtDepth(10, bmin, bmax);
	
	std::vector<float> bmin_1, bmax_1;
	std::vector<int> depths;
	kdtree.getNodesBoundingBoxAtMaxDepth(10, bmin, bmax, depths);

	PointXYZ p_leaf;
	for (int i = 0; i < 3; ++i)
		p_leaf.data[i] = 0.5 * (bmin[i] + bmax[i]);
	std::vector<int> leaf_point_indices;
	kdtree.getLeafPointIndices(p_leaf, leaf_point_indices);

	char pause;
	std::cout << "press any key to escape...";
	std::cin >> pause; 
	
	return 0;
}
