#ifndef OH_OBJ_LOADER_H
#define OH_OBJ_LOADER_H

#include <string>
#include "pcl/point_cloud.h"
#include "pcl/point_types.h"
namespace pcl
{
	namespace io
	{
		PCL_EXPORTS bool cloudload_point_cloud(const std::string &filename, const std::string &basepath, PointCloud<PointXYZ>::Ptr &cloud);
	}
}
#endif
