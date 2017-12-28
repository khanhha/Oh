#ifndef PCL_FILTERS_UNIFORM_OCTREE_SAMPLING_IMPL_H_
#define PCL_FILTERS_UNIFORM_OCTREE_SAMPLING_IMPL_H_


using namespace std;
using namespace pcl;
using namespace Eigen;

template <typename PointT>
void pcl::UniformOctreeSampling<PointT>::applyFilter(PointCloud &output)
{
	octree_.reset(new OctreeNormal(octree_resolution_));
	octree_->setInputCloud(input_);
	octree_->enableDynamicDepth(50);
	octree_->setInputNormalCloud(input_normal_cloud_);
	octree_->setNormalThreshold(octree_normal_threshold);
	octree_->addPointsFromInputCloud();

	Vector3d bmin, bmax;
	octree_->getBoundingBox(bmin[0], bmin[1], bmin[1], bmax[0], bmax[1], bmax[2]);
	initSamplingBounds(bmin.cast<float>(), bmax.cast<float>());

	bool no_more = false;
	vector<float> radius_heights;
	vector<float> radius_sqr_dsts;
	radius_heights.reserve(20);
	radius_sqr_dsts.reserve(20);

	OctreeNormal::LeafNodeIterator leafIter, leafEnd = octree_->leaf_end();
	size_t test_cnt = 0;
	for (leafIter = octree_->leaf_begin(); leafIter != leafEnd; ++leafIter, test_cnt++)
	{
		//if(test_cnt != 8634)
		//	continue;
		OctreeNormal::LeafNode *leaf = static_cast<OctreeNormal::LeafNode*>(*leafIter);
		octree::OctreeContainerPointIndices *container = static_cast<octree::OctreeContainerPointIndices*>(leaf->getContainerPtr());
		const std::vector<int> &indices = container->getPointIndicesVector();

		Vector3f node_bmin, node_bmax;
#if 1
		octree_->getVoxelBounds(leafIter, node_bmin, node_bmax);
#else
		calcBounds(indices, node_bmin, node_bmax);
#endif

//#define	TEST_VOXEL_BOUNDS
#ifdef TEST_VOXEL_BOUNDS
		Vector3f voxel_min, voxel_max;
		octree_->getVoxelBounds(leafIter, voxel_min, voxel_max);
		test_node_bounds.push_back(std::make_pair(voxel_min, voxel_max));
#endif

		//test_node_bounds.push_back(std::make_pair(node_bmin, node_bmax));
		//test_node_ids.push_back(std::make_pair(0.5f*(node_bmin + node_bmax), test_cnt));

		for (size_t i = 0; i < indices.size(); ++i)
			test_node_points.push_back(input_->points[indices[i]].getVector3fMap());

		Vector3f snode_bmin = inverse_sampling_size_.array()*node_bmin.array();
		Vector3f snode_bmax = inverse_sampling_size_.array()*node_bmax.array();
		Vector3f node_bmin_i(ceil(snode_bmin[0]),  ceil(snode_bmin[1]), ceil(snode_bmin[2]));
		Vector3f node_bmax_i(floor(snode_bmax[0]), floor(snode_bmax[1]), floor(snode_bmax[2]));
		
		Vector3f node_avg_norm, node_avg_center;
		averagePlane(leaf, node_avg_center, node_avg_norm);
		const int base_plane_norm_axis = findBasePlane(node_avg_norm);
		const int u_axis = (base_plane_norm_axis + 1) % 3;
		const int v_axis = (base_plane_norm_axis + 2) % 3;
		const int width =  node_bmax_i[u_axis] - node_bmin_i[u_axis];
		const int height = node_bmax_i[v_axis] - node_bmin_i[v_axis];
		node_bmin_i *= sampling_resolution_;
		node_bmax_i *= sampling_resolution_;

		Vector3f test_size = node_bmax_i - node_bmin_i;
		if (test_size[u_axis] < 0.0f && test_size[v_axis] < 0.0f)
			continue;

		size_t inside_bounds_cnt = 0;
		for (int i = 0; i <= width; ++i) {
			for (int j = 0; j <= height; ++j) {
				Vector3f sampled_co;
				sampled_co[u_axis] = node_bmin_i[u_axis] + i * sampling_resolution_;
				sampled_co[v_axis] = node_bmin_i[v_axis] + j * sampling_resolution_;
				
				radius_heights.clear(); radius_sqr_dsts.clear();
				bool inside_bounds = searchRadiusOnPlane(node_avg_norm, node_avg_center, 
					sampled_co, sample_radius_search, u_axis, v_axis, base_plane_norm_axis,
					node_bmin, node_bmax, radius_heights, radius_sqr_dsts);
				
				if(!inside_bounds)
					continue;

				inside_bounds_cnt++;

				//test_sample_points_1.push_back(sampled_co);

				//interpolate
				float total_weight = 0.0f;
				float h = 0.0f;
				for (size_t k = 0; k < radius_heights.size(); ++k)
				{
					const float w = (radius_sqr_dsts[k] != 0.0f) ? (1.0f / sqrt(radius_sqr_dsts[k])): 1.0f;
					const float h_tmp = radius_heights[k];
					//assert(h_tmp >= 0.0f);
					total_weight += w;
					h += h_tmp * w;
				}

				h = h / total_weight;

				sampled_co[base_plane_norm_axis] = node_bmin[base_plane_norm_axis] + h;

				test_sample_points.push_back(sampled_co);
			}
		}

		float percent = static_cast<float>(inside_bounds_cnt) / static_cast<float>((width+1)*(height+1));
		if (percent < 0.9)
		{
			test_node_bounds.push_back(std::make_pair(node_bmin, node_bmax));
		}
	}
}

inline bool nearBoundary(const size_t &u, const size_t &v, const Vector3f &min_corner, const Vector3f &max_corner, const float &radius, const Vector3f &p)
{
	if ((p[u] - min_corner[u] >= 0.0f && p[u] - min_corner[u] < radius) ||
		(p[v] - min_corner[v] >= 0.0f && p[v] - min_corner[v] < radius) ||
		(max_corner[u] - p[u] >= 0.0f && max_corner[u] - p[u] < radius) ||
		(max_corner[v] - p[v] >= 0.0f && max_corner[v] - p[v] < radius))
	{
		return true;
	}
	return false;
}

inline Vector3f extendBoundary(const size_t &u, const size_t &v, const Vector3f &min_corner, const Vector3f &max_corner, const float &radius, const Vector3f &p)
{
	Vector3f size = max_corner - min_corner;
	Vector3f pr = p - min_corner;
	const float dia_ratio = size[v] / size[u];
	const float ratio = pr[v] / pr[u];
	if (ratio > dia_ratio)
	{
		if (pr[u] < radius)
		{
			pr[u] = -pr[u];
		}
		else if (size[v] - pr[v] < radius)
		{
			pr[v] += 2 * (size[v] - pr[v]);
		}
	}
	else if (ratio < dia_ratio)
	{
		if (pr[v] < radius)
		{
			pr[v] = -pr[v];
		}
		else if (size[u] - pr[u] < radius)
		{
			pr[u] += 2 * (size[u] - pr[u]);
		}
	}
	else 
	{
		assert(false);
	}

	return min_corner + pr;
}

inline float squareDistance(const Vector3f &p0, const Vector3f &p1, const size_t &u, const size_t &v)
{
	const float du = p0[u] - p1[u];
	const float dv = p0[v] - p1[v];
	return (du * du + dv * dv);
}

inline bool planeRayInteraction(const Eigen::Vector3f &plane_n, const Eigen::Vector3f &plane_p, const Eigen::Vector3f &org, const Vector3f &dir, Vector3f &inter)
{
	float denom = dir.dot(plane_n);
	if (denom != 0.0f) {
		float t = (plane_p - org).dot(plane_n) / denom;
		inter = org + t * dir;
		return true;
	}
	else {
		return false;
	}
}

template <typename PointT>
bool pcl::UniformOctreeSampling<PointT>::searchRadiusOnPlane(
	const Eigen::Vector3f &plane_n, const Eigen::Vector3f &plane_p,
	const Vector3f &search_p, float radius, size_t u, size_t v, size_t height_axis,
	const Vector3f &leaf_bmin, const Vector3f &leaf_bmax,
	std::vector<float> &ret_heights, std::vector<float> &ret_sqrt_dst)
{
	Vector3f on_lower_plane = search_p;
	on_lower_plane[height_axis] = leaf_bmin[height_axis];
	Vector3f dir = Vector3f::Zero();
	dir[height_axis] = 1.0f;
	
	Vector3f search_p_3d;
	assert(planeRayInteraction(plane_n, plane_p, on_lower_plane, dir, search_p_3d));

	float intersect_height = search_p_3d[height_axis] - leaf_bmin[height_axis];
	if (intersect_height < 0 || intersect_height > (leaf_bmax[height_axis] - leaf_bmin[height_axis]))
		return false;

	//test_sample_points_2.push_back(search_p_3d);

	PointT p; p.x = search_p_3d[0]; p.y = search_p_3d[1]; p.z = search_p_3d[2];
	
	std::vector<int> indices_3d;
	std::vector<float> dst_3d;
	int npoints = octree_->radiusSearch(p, radius, indices_3d, dst_3d);
	ret_heights.resize(npoints);
	ret_sqrt_dst.resize(npoints);
	for (int i = 0; i < npoints ; ++i)
	{
		const Vector3f tmp = input_->points[indices_3d[i]].getVector3fMap();
		ret_heights[i]  = tmp[height_axis] - leaf_bmin[height_axis];
		ret_sqrt_dst[i] = squareDistance(search_p_3d, tmp, u, v);
	}

	return true;
}

template <typename PointT>
size_t pcl::UniformOctreeSampling<PointT>::findBasePlane(const Eigen::Vector3f &avg_norm) const
{
	size_t max_axis = 0;
	float  max_dot = -10.0f;
	for (size_t i = 0; i < 3; ++i) 
	{
		Vector3f axis = Vector3f::Zero(); axis[i] = 1.0f;
		float dot_tmp = std::abs(avg_norm.dot(axis));
		if (dot_tmp > max_dot) {
			max_axis = i;
			max_dot = dot_tmp;
		}
	}

	return max_axis;
}

template <typename PointT>
void pcl::UniformOctreeSampling<PointT>::averagePlane(const LeafNode *leaf, Eigen::Vector3f &p, Eigen::Vector3f &n) const
{
	const octree::OctreeContainerPointIndices *container = static_cast<const octree::OctreeContainerPointIndices*>(leaf->getContainerPtr());
	const std::vector<int>& indices = container->getPointIndicesVector();
	Eigen::Vector3f avg_norm = Vector3f::Zero();
	Eigen::Vector3f avg = Vector3f::Zero();
	for (auto it = indices.cbegin(); it != indices.cend(); ++it)
	{
		const Eigen::Vector3f& norm = input_normal_cloud_->at(*it).getNormalVector3fMap();
		avg_norm += norm;
		avg += input_->points[*it].getVector3fMap();
	}
	avg_norm /= indices.size();
	avg_norm.normalize();

	p = avg /indices.size();
	n = avg_norm;
}

template <typename PointT>
void pcl::UniformOctreeSampling<PointT>::calcBounds(const std::vector<int> &indices, Eigen::Vector3f &bmin, Eigen::Vector3f &bmax)
{
	typename std::vector<int>::const_iterator it;
	typename std::vector<int>::const_iterator it_end = indices.cend();

	bmin[0] = bmin[1] = bmin[2] =  std::numeric_limits <float>::max();
	bmax[0] = bmax[1] = bmax[2] = -std::numeric_limits <float>::max();

	for (it = indices.cbegin(); it != it_end; ++it)
	{
		const Eigen::Vector3f& p = input_->at(*it).getVector3fMap();
		bmin = bmin.array().min(p.array());
		bmax = bmax.array().max(p.array());
	}
}
#endif