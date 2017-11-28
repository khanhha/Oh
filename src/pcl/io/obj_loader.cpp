#include "obj_loader.h"
#include "tiny_obj_loader.h"

bool pcl::io::cloudload_point_cloud(const std::string &filename, const std::string &basepath, PointCloud<PointXYZ>::Ptr &cloud)
{
	tinyobj::attrib_t attrib;
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;
	std::string err;

	bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &err, filename.c_str(), basepath.c_str(), false);

	if (ret)
	{
		cloud = PointCloud<PointXYZ>::Ptr(new PointCloud<PointXYZ>());

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
		return true;
	}
	else {
		cloud = PointCloud<PointXYZ>::Ptr();
		return false;
	}
}

