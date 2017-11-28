#include <string>
#include <vector>
#include "point_cloud.h"
#include "point_types.h"
//#define TINYOBJLOADER_IMPLEMENTATION
#include "io/tiny_obj_loader.h"

using namespace std;
using namespace pcl;

PointCloud<PointXYZ>::Ptr load_pcl(const string &filename, const string &basepath)
{
	tinyobj::attrib_t attrib;
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;
	std::string err;

	bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &err, filename.c_str(), basepath.c_str(), false);

	if (ret)
	{
		PointCloud<PointXYZ>::Ptr cloud = PointCloud<PointXYZ>::Ptr(new PointCloud<PointXYZ>());

		for (size_t v = 0; v < attrib.vertices.size() / 3; v++) {
			PointXYZ p(attrib.vertices[3 * v + 0], attrib.vertices[3 * v + 1], attrib.vertices[3 * v + 2]);
			cloud->push_back(p);
#if 0
			printf("  v[%ld] = (%f, %f, %f)\n", static_cast<long>(v),
				static_cast<const double>(attrib.vertices[3 * v + 0]),
				static_cast<const double>(attrib.vertices[3 * v + 1]),
				static_cast<const double>(attrib.vertices[3 * v + 2]));
#endif
		}

		return cloud;
	}
	else {
		return  PointCloud<PointXYZ>::Ptr();
	}
}

int main()
{
	string filename = "lucy_none-Slice-54_center_vn.obj";
	string basepath = "G:\\Projects\\Oh\\data\\test_data\\";
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = load_pcl(basepath+filename, basepath);

	return 0;
}