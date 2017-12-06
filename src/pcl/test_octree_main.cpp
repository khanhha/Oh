#include <string>
#include <vector>
#include "point_cloud.h"
#include "point_types.h"
//#define TINYOBJLOADER_IMPLEMENTATION
#include "io/tiny_obj_loader.h"
#include <pcl/octree/octree.h>
#include <random>

using namespace std;
using namespace pcl;

void load_pcl(const string &filename, const string &basepath, PointCloud<PointXYZ>::Ptr cloud, PointCloud<Normal>::Ptr normal)
{
	tinyobj::attrib_t attrib;
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;
	std::string err;

	bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &err, filename.c_str(),  basepath.c_str(), false);

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
	string filename = "normal_lucy_none-Slice-55_center_vn.obj";
	string basepath = "G:\\Projects\\Oh\\data\\test_data\\";

	PointCloud<PointXYZ>::Ptr cloud = PointCloud<PointXYZ>::Ptr(new PointCloud<PointXYZ>());
	PointCloud<Normal>::Ptr	normal	= PointCloud<Normal>::Ptr(new PointCloud<Normal>());

	load_pcl(basepath+filename, basepath, cloud, normal);

#if 0
	pcl::octree::OctreePointCloudSearch<pcl::PointXYZ> oc(0.1);
	oc.setInputCloud(cloud);
	oc.addPointsFromInputCloud();
#else
	pcl::octree::OctreePointCloudNormal<pcl::PointXYZ, pcl::Normal> oc(0.1);
	oc.setInputCloud(cloud);
	oc.setInputNormalCloud(normal);
	//octree.setNormalThreshold(0.8);
	oc.enableDynamicDepth(100);
	oc.addPointsFromInputCloud();
#endif

	std::vector<int> idxs;
	std::vector<float> dsts;
	oc.radiusSearch(50, 50.0, idxs, dsts, 100);
	
	idxs.clear(); dsts.clear();
	oc.nearestKSearch(50, 50, idxs, dsts);

	int ret_idx; float ret_dst;
	oc.approxNearestSearch(5, ret_idx, ret_dst);

	idxs.clear();
	PointXYZ bmin(0, 0, 0);
	PointXYZ bmax(500, 500, 500);
	oc.boxSearch(bmin, bmax, idxs);

	std::vector<octree::OctreeKey> keys;
	std::vector<int> depths;
	oc.getAllLeafKeys(keys, depths);
	for (auto it = keys.begin(); it != keys.end(); ++it)
	{
		octree::OctreeKey key = *it;
		auto leafnode = oc.findLeaf(key.x, key.y, key.z);
		size_t npoint = leafnode->getSize();
		if (npoint == 0) {
			int somethinewrong = 1;
		}
	}
	
	std::vector<octree::OctreeKey> keys_1;
	std::vector<int> depths_1;
	oc.getOctreeKeysAtMaxDepth(4, keys_1, depths_1);

	return 0;
}