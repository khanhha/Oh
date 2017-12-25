#ifndef PCL_FILTERS_UNIFORM_OCTREE_SAMPLING_IMPL_H_
#define PCL_FILTERS_UNIFORM_OCTREE_SAMPLING_IMPL_H_


using namespace std;
using namespace pcl;
using namespace Eigen;

template <typename PointT>
void pcl::UniformOctreeSampling<PointT>::applyFilter(PointCloud &output)
{
	_octree.reset(new OctreeNormal(octree_resolution_));
	_octree->setInputCloud(input_);
	_octree->enableDynamicDepth(50);
	_octree->setInputNormalCloud(input_normal_cloud_);
	_octree->setNormalThreshold(octree_normal_threshold);
	_octree->addPointsFromInputCloud();

	Vector3d bmin, bmax;
	_octree->getBoundingBox(bmin[0], bmin[1], bmin[1], bmax[0], bmax[1], bmax[2]);
	initSamplingBounds(bmin.cast<float>(), bmax.cast<float>());

	bool no_more = false;
	vector<Vector3f> radius_points;
	vector<float> radius_sqr_dsts;
	radius_points.reserve(20);
	radius_sqr_dsts.reserve(20);

	OctreeNormal::LeafNodeIterator leafIter, leafEnd = _octree->leaf_end();
	for (leafIter = _octree->leaf_begin(); leafIter != leafEnd; ++leafIter)
	{
		OctreeNormal::LeafNode *leaf = static_cast<OctreeNormal::LeafNode*>(*leafIter);
		octree::OctreeContainerPointIndices *container = static_cast<octree::OctreeContainerPointIndices*>(leaf->getContainerPtr());
		const std::vector<int> &indices = container->getPointIndicesVector();

		Vector3f node_bmin, node_bmax;
#if 0
		oc.getVoxelBounds(leafIter, node_bmin, node_bmax);
#else
		calcBounds(indices, node_bmin, node_bmax);
#endif
		test_node_bounds.push_back(std::make_pair(node_bmin, node_bmax));

		//if (no_more)
		//	return;
		if(indices.size() < 10)
			continue;
		//no_more = true;

		for (size_t i = 0; i < indices.size(); ++i)
			test_node_points.push_back(input_->points[indices[i]].getVector3fMap());
		

		//Vector3i node_bmin_i = mapSamplingCoord(node_bmin);
		//Vector3i node_bmax_i = mapSamplingCoord(node_bmax);
		Vector3f snode_bmin = inverse_sampling_size_.array()*node_bmin.array();
		Vector3f snode_bmax = inverse_sampling_size_.array()*node_bmax.array();
		Vector3f node_bmin_i(ceil(snode_bmin[0]),  ceil(snode_bmin[1]), ceil(snode_bmin[2]));
		Vector3f node_bmax_i(floor(snode_bmax[0]), floor(snode_bmax[1]), floor(snode_bmax[2]));
		
		const int base_plane_norm_axis = findBasePlane(leaf);
		const int u_axis = (base_plane_norm_axis + 1) % 3;
		const int v_axis = (base_plane_norm_axis + 2) % 3;
		const int width =  node_bmax_i[u_axis] - node_bmin_i[u_axis];
		const int height = node_bmax_i[v_axis] - node_bmin_i[v_axis];
		node_bmin_i *= sampling_resolution_;
		node_bmax_i *= sampling_resolution_;

		Vector3f test_size = node_bmax_i - node_bmin_i;
		if (test_size[u_axis] == 0.0f || test_size[v_axis] == 0.0f)
		{
			continue;
		}

		for (int i = 0; i <= width; ++i) {
			for (int j = 0; j <= height; ++j) {
				Vector3f sampled_co;
				sampled_co[u_axis] = node_bmin_i[u_axis] + i * sampling_resolution_;
				sampled_co[v_axis] = node_bmin_i[v_axis] + j * sampling_resolution_;
				
				radius_points.clear(); radius_sqr_dsts.clear();
				searchRadiusOnPlane(indices, sampled_co, sample_radius_search, u_axis, v_axis, base_plane_norm_axis,
					node_bmin_i, node_bmax_i, node_bmin, radius_points, radius_sqr_dsts);

				if (radius_points.size() < 4)
				{
					//sampled_co[base_plane_norm_axis] = 0.5f * (node_bmin[base_plane_norm_axis] + node_bmax[base_plane_norm_axis]);
					//test_sample_points_1.push_back(sampled_co);
					continue;
				}

				//interpolate
				float total_weight = 0.0f;
				float h = 0.0f;
				for (size_t k = 0; k < radius_points.size(); ++k)
				{
					const float w = (radius_sqr_dsts[k] != 0.0f) ? (1.0f / sqrt(radius_sqr_dsts[k])): 1.0f;
					const float h_tmp = radius_points[k][base_plane_norm_axis];
					//assert(h_tmp >= 0.0f);
					total_weight += w;
					h += h_tmp * w;
				}

				h = h / total_weight;

				sampled_co[base_plane_norm_axis] = node_bmin[base_plane_norm_axis] + h;

				test_sample_points.push_back(sampled_co);
			}
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

template <typename PointT>
size_t pcl::UniformOctreeSampling<PointT>::searchRadiusOnPlane(const std::vector<int> &all_indices, 
	const Vector3f &search_p, float radius, size_t u, size_t v, size_t height_axis,
	const Vector3f &sample_bmin, const Vector3f &sample_bmax, const Vector3f &leaf_bmin,
	std::vector<Vector3f> &ret_vertices, std::vector<float> &ret_sqrt_dst) const
{
	float sqr_radius = radius * radius;
#if 0
	bool  search_near_boundary = nearBoundary(u, v, sample_bmin, sample_bmax, radius, search_p);

	for (int i = 0; i < all_indices.size(); ++i)
	{
		Vector3f p = input_->points[all_indices[i]].getVector3fMap();
		p[height_axis] =  p[height_axis] - leaf_bmin[height_axis];

		float sqr_dst = squareDistance(p, search_p, u, v);
		if (sqr_dst <= sqr_radius)
		{
			ret_vertices.push_back(p);
			ret_sqrt_dst.push_back(sqr_dst);

			if (search_near_boundary && nearBoundary(u, v, sample_bmin, sample_bmax, radius, p))
			{
				Vector3f flip = extendBoundary(u, v, sample_bmin, sample_bmax, radius, p);
				flip[height_axis] = p[height_axis];
				ret_vertices.push_back(flip);
				ret_sqrt_dst.push_back(squareDistance(flip, search_p, u, v));
			}
		}
	}
#else

#endif
	return ret_vertices.size();
}

template <typename PointT>
size_t pcl::UniformOctreeSampling<PointT>::findBasePlane(const LeafNode *leaf) const
{
	const octree::OctreeContainerPointIndices *container = static_cast<const octree::OctreeContainerPointIndices*>(leaf->getContainerPtr());
	const std::vector<int>& indices = container->getPointIndicesVector();
	Eigen::Vector3f avg_norm = Eigen::Vector3f::Zero();
	for (auto it = indices.cbegin(); it != indices.cend(); ++it)
	{
		const Eigen::Vector3f& norm = input_normal_cloud_->at(*it).getNormalVector3fMap();
		avg_norm += norm;
	}
	avg_norm /= indices.size();
	avg_norm.normalize();

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