#ifndef PCL_FILTERS_UNIFORM_OCTREE_SAMPLING_IMPL_H_
#define PCL_FILTERS_UNIFORM_OCTREE_SAMPLING_IMPL_H_

#include <pcl/octree/octree_pointcloud_normal.h>

using namespace std;
using namespace pcl;
using namespace Eigen;

template <typename PointT>
void pcl::UniformOctreeSampling<PointT>::applyFilter(PointCloud &output)
{
	OctreeNormal oc(octree_resolution_);
	oc.setInputCloud(input_);
	oc.enableDynamicDepth(50);
	oc.setInputNormalCloud(input_normal_cloud_);
	oc.setNormalThreshold(octree_normal_threshold);
	oc.addPointsFromInputCloud();

	Vector3d bmin, bmax;
	oc.getBoundingBox(bmin[0], bmin[1], bmin[1], bmax[0], bmax[1], bmax[2]);
	initSamplingBounds(bmin.cast<float>(), bmax.cast<float>());

	bool no_more = false;

	OctreeNormal::LeafNodeIterator leafIter, leafEnd = oc.leaf_end();
	for (leafIter = oc.leaf_begin(); leafIter != leafEnd; ++leafIter)
	{
		OctreeNormal::LeafNode *leaf = static_cast<OctreeNormal::LeafNode*>(*leafIter);
		octree::OctreeContainerPointIndices *container = static_cast<octree::OctreeContainerPointIndices*>(leaf->getContainerPtr());
		const std::vector<int> &indices = container->getPointIndicesVector();

		if (no_more)
			return;
		
		if(indices.size() < 50)
			continue;

		no_more = true;

		Vector3f node_bmin, node_bmax;
		oc.getVoxelBounds(leafIter, node_bmin, node_bmax);

		for (size_t i = 0; i < indices.size(); ++i)
			test_node_points.push_back(input_->points[indices[i]].getVector3fMap());
		
		test_node_bounds.push_back(std::make_pair(node_bmin, node_bmax));

		Vector3i node_bmin_i = mapSamplingCoord(node_bmin);
		Vector3i node_bmax_i = mapSamplingCoord(node_bmax);
		const size_t base_plane_norm_axis = findBasePlane(leaf);
		const size_t u_axis = (base_plane_norm_axis + 1) % 3;
		const size_t v_axis = (base_plane_norm_axis + 2) % 3;
		const size_t width =  node_bmax_i[u_axis] - node_bmin_i[u_axis];
		const size_t height = node_bmax_i[v_axis] - node_bmin_i[v_axis];
		for (size_t i = 0; i < width; ++i) {
			for (size_t j = 0; j < height; ++j) {
				Vector3i sam_co;
				sam_co[u_axis] = node_bmin_i[u_axis] + i;
				sam_co[v_axis] = node_bmin_i[v_axis] + j;

				Vector3f sampled_co;
				sampled_co[u_axis] = sam_co[u_axis] * sampling_resolution_;
				sampled_co[v_axis] = sam_co[v_axis] * sampling_resolution_;
				
				vector<int> radius_indices;
				vector<float> radius_sqr_dsts;
				searchRadiusOnPlane(indices, sampled_co, sample_radius_search * sample_radius_search, u_axis, v_axis, radius_indices, radius_sqr_dsts);

				if (radius_indices.empty())
				{
					sampled_co[base_plane_norm_axis] = 0.5f * (node_bmin[base_plane_norm_axis] + node_bmax[base_plane_norm_axis]);
					test_sample_points_1.push_back(sampled_co);
					continue;
				}

				//interpolate
				float total_weight = 0.0f;
				float h = 0.0f;
				for (size_t k = 0; k < radius_indices.size(); ++k)
				{
					float w = 1.0f / sqrt(radius_sqr_dsts[k]);
					float h_tmp = heightBasePlane(indices[radius_indices[k]], base_plane_norm_axis, node_bmin);
					assert(h_tmp >= 0.0f);
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

template <typename PointT>
size_t pcl::UniformOctreeSampling<PointT>::searchRadiusOnPlane(const std::vector<int> &all_indices, const Vector3f &search_p, float sqr_radius, size_t u, size_t v, std::vector<int> &ret_indices, std::vector<float> &ret_sqrt_dst) const
{
	for (int i = 0; i < all_indices.size(); ++i)
	{
		const Vector3f &p = input_->points[all_indices[i]].getVector3fMap();
		const float du = p[u] - search_p[u]; 
		const float dv = p[v] - search_p[v];
		float sqr_dst = du * du + dv * dv;
		if (sqr_dst <= sqr_radius)
		{
			ret_indices.push_back(i);
			ret_sqrt_dst.push_back(sqr_dst);
		}
	}

	return ret_indices.size();
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
#endif