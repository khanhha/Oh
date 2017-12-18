#ifndef OH_OBJ_LOADER_H
#define OH_OBJ_LOADER_H

#include <string>
#include "pcl/point_cloud.h"
#include "pcl/point_types.h"
namespace pcl
{
	namespace io
	{
		PCL_EXPORTS bool cloud_load_point_cloud(const std::string &filename, const std::string &basepath, PointCloud<PointXYZ>::Ptr &cloud);
		PCL_EXPORTS bool cloud_load_point_cloud(const std::string &filename, const std::string &basepath, PointCloud<PointXYZ>::Ptr &cloud, PointCloud<Normal>::Ptr &ncloud);
	}
}
#endif
