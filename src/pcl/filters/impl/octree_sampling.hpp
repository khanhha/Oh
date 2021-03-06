#ifndef PCL_FILTERS_UNIFORM_OCTREE_SAMPLING_IMPL_H_
#define PCL_FILTERS_UNIFORM_OCTREE_SAMPLING_IMPL_H_


using namespace std;
using namespace pcl;
using namespace Eigen;


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

//adapted from  http://geomalgorithms.com/a03-_inclusion.html
// cn_PnPoly(): crossing number test for a point in a polygon
//      Input:   P = a point,
//               V[] = vertex points of a polygon V[n+1] with V[n]=V[0]
//      Return:  0 = outside, 1 = inside
// This code is patterned after [Franklin, 2000]
inline int IsPointInsidePoly(const Vector3f &P, const std::vector<Vector3f> &V, const size_t &n, const size_t &u, const size_t &v)
{
	size_t    cn = 0;    // the  crossing number counter
						 // loop through all edges of the polygon
	for (int i = 0; i < n; i++) {    // edge from V[i]  to V[i+1]
		if (((V[i][v] <= P[v]) && (V[i + 1][v] > P[v]))     // an upward crossing
			|| ((V[i][v] > P[v]) && (V[i + 1][v] <= P[v]))) { // a downward crossing
															  // compute  the actual edge-ray intersect x-coordinate
			float vt = (float)(P[v] - V[i][v]) / (V[i + 1][v] - V[i][v]);
			if (P[u] < V[i][u] + vt * (V[i + 1][u] - V[i][u])) // P[u] < intersect
				++cn;   // a valid crossing of y=P.y right of P.x
		}
	}

	return (cn & 1);    // 0 if even (out), and 1 if  odd (in)
}

// 2D cross product of OA and OB vectors, i.e. z-component of their 3D cross product.
// Returns a positive value, if OAB makes a counter-clockwise turn,
// negative for clockwise turn, and zero if the points are collinear.
inline float cross(const Vector3f &O, const Vector3f &A, const Vector3f &B, size_t u, size_t v)
{
	return (A[u] - O[u]) * (B[v] - O[v]) - (A[v] - O[v]) * (B[u] - O[u]);
}

// Returns a list of points on the convex hull in counter-clockwise order.
// Note: the last point in the returned list is the same as the first one.
size_t buildConvexHull(vector<Vector3f> &P, std::vector<Vector3f> &H, size_t u, size_t v)
{
	int n = P.size(), k = 0;
	if (n == 1) return P.size();
	H.resize(2 * n);

	// Sort points lexicographically
	std::sort(P.begin(), P.end(), [&](const Vector3f &a, const Vector3f &b) {return a[u] < b[u] || (a[u] == b[u] && a[v] < b[v]); });

	// Build lower hull
	for (int i = 0; i < n; ++i) {
		while (k >= 2 && cross(H[k - 2], H[k - 1], P[i], u, v) <= 0) k--;
		H[k++] = P[i];
	}

	// Build upper hull
	for (int i = n - 2, t = k + 1; i >= 0; i--) {
		while (k >= t && cross(H[k - 2], H[k - 1], P[i], u, v) <= 0) k--;
		H[k++] = P[i];
	}

	H.resize(k - 1);

	return H.size();
}

template <typename PointT>
void pcl::OctreeSampling<PointT>::applyFilter(PointCloud &output)
{
	output.height = 1;                    // down sampling breaks the organized structure
	output.is_dense = true;                 // we filter out invalid points

	if (method_ == ResampleMethod::UNIFORM)
		filterUniform(output);
	else if (method_ == ResampleMethod::NONUNIFORM_MAX_POINTS_PER_LEAF)
		filterNonuniformMaxPointsPerLeaf(output);
	else if (method_ == ResampleMethod::NONUNIFORM_NORMAL_THRESHOLD)
		filterNonuniformNormalThreshold(output);
}

template <typename PointT>
void pcl::OctreeSampling<PointT>::filterUniform(PointCloud &output)
{
	typedef octree::OctreePointCloud<PointT> MyOctree;
	MyOctree oc(sampling_resolution_);
	oc.setInputCloud(this->input_);
	oc.addPointsFromInputCloud();

	typename MyOctree::LeafNodeIterator leafIter, leafEnd = oc.leaf_end();
	for (leafIter = oc.leaf_begin(); leafIter != leafEnd; ++leafIter)
	{
		typename  MyOctree::LeafNode *leaf = static_cast<typename MyOctree::LeafNode*>(*leafIter);
		octree::OctreeContainerPointIndices *container = static_cast<octree::OctreeContainerPointIndices*>(leaf->getContainerPtr());
		const std::vector<int> &indices = container->getPointIndicesVector();

		PointT sample_point;
		if (interpolation_ == InterpolationMethod::CLOSEST_TO_CENTER)
		{
			PointT p;
			oc.genLeafNodeCenterFromOctreeKey(leafIter.getCurrentOctreeKey(), p);
			Vector3f leaf_center(p.x, p.y, p.z);
			closestPoint(indices, leaf_center, sample_point);
		}
		else if (interpolation_ == InterpolationMethod::AVERAGE)
		{
			averagePoint(indices, sample_point);
		}
		else
		{
			heightApproximate(indices, sample_point);
		}

		output.points.push_back(sample_point);
	}
}

template <typename PointT>
void pcl::OctreeSampling<PointT>::filterNonuniformMaxPointsPerLeaf(PointCloud &output)
{
	typedef octree::OctreePointCloud<PointT> MyOctree;
	MyOctree oc(octree_resolution_);
	oc.setInputCloud(this->input_);
	oc.enableDynamicDepth(max_points_per_octree_leaf_);
	oc.addPointsFromInputCloud();

	typename MyOctree::LeafNodeIterator leafIter, leafEnd = oc.leaf_end();
	for (leafIter = oc.leaf_begin(); leafIter != leafEnd; ++leafIter)
	{
		typename MyOctree::LeafNode *leaf = static_cast<typename MyOctree::LeafNode*>(*leafIter);
		octree::OctreeContainerPointIndices *container = static_cast<octree::OctreeContainerPointIndices*>(leaf->getContainerPtr());
		const std::vector<int> &indices = container->getPointIndicesVector();

		PointT sample_point;
		if (interpolation_ == InterpolationMethod::CLOSEST_TO_CENTER)
		{
			PointT p;
			oc.genLeafNodeCenterFromOctreeKey(leafIter.getCurrentOctreeKey(), p);
			Vector3f leaf_center(p.x, p.y, p.z);
			closestPoint(indices, leaf_center, sample_point);
		}
		else if (interpolation_ == InterpolationMethod::AVERAGE)
		{
			averagePoint(indices, sample_point);
		}
		else 
		{
			heightApproximate(indices, sample_point);
		}

		output.points.push_back(sample_point);
	}
}

template <typename PointT>
void pcl::OctreeSampling<PointT>::filterNonuniformNormalThreshold(PointCloud &output)
{
	octree_.reset(new OctreeNormal(octree_resolution_));
	octree_->setInputCloud(this->input_);
	octree_->enableDynamicDepth(64);
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

	typename OctreeNormal::LeafNodeIterator leafIter, leafEnd = octree_->leaf_end();
	size_t test_cnt = 0;
	for (leafIter = octree_->leaf_begin(); leafIter != leafEnd; ++leafIter, test_cnt++)
	{
		typename OctreeNormal::LeafNode *leaf = static_cast<typename OctreeNormal::LeafNode*>(*leafIter);
		octree::OctreeContainerPointIndices *container = static_cast<octree::OctreeContainerPointIndices*>(leaf->getContainerPtr());
		const std::vector<int> &indices = container->getPointIndicesVector();

		Vector3f voxel_bmin, voxel_bmax;
		octree_->getVoxelBounds(leafIter, voxel_bmin, voxel_bmax);
		Vector3f node_bmin, node_bmax;
		calcBounds(indices, node_bmin, node_bmax);

		//test_node_bounds.push_back(std::make_pair(voxel_bmin, voxel_bmax));
		//test_node_ids.push_back(std::make_pair(0.5f*(voxel_bmin + voxel_bmax), test_cnt));
		//for (size_t i = 0; i < indices.size(); ++i)
		//	test_node_points.push_back(input_->points[indices[i]].getVector3fMap());

		Vector3f snode_bmin = inverse_sampling_size_.array()*voxel_bmin.array();
		Vector3f snode_bmax = inverse_sampling_size_.array()*voxel_bmax.array();
		Vector3f node_bmin_i(ceil(snode_bmin[0]), ceil(snode_bmin[1]), ceil(snode_bmin[2]));
		Vector3f node_bmax_i(floor(snode_bmax[0]), floor(snode_bmax[1]), floor(snode_bmax[2]));

		Vector3f node_avg_norm, node_avg_center;
		averagePlane(leaf, node_avg_center, node_avg_norm);
		const int base_plane_norm_axis = findBasePlane(node_avg_norm);
		const int u_axis = (base_plane_norm_axis + 1) % 3;
		const int v_axis = (base_plane_norm_axis + 2) % 3;
		const int width = node_bmax_i[u_axis] - node_bmin_i[u_axis];
		const int height = node_bmax_i[v_axis] - node_bmin_i[v_axis];
		node_bmin_i *= sampling_resolution_;
		node_bmax_i *= sampling_resolution_;
		float voxel_height = voxel_bmax[base_plane_norm_axis] - voxel_bmin[base_plane_norm_axis];

		Vector3f test_size = node_bmax_i - node_bmin_i;
		if (test_size[u_axis] < 0.0f && test_size[v_axis] < 0.0f)
			continue;

		if (indices.size() < 3)
			continue;

		std::vector<Vector3f> extend_leaf_points;
		Vector3f leaf_center = 0.5f*(node_bmin + node_bmax);
		float radius_search = 0.5*(node_bmax - node_bmin).norm();
		radius_search = radius_search + 1.1* sampling_resolution_;
		searchPointsRadius(leaf_center, radius_search, &extend_leaf_points, nullptr);

		std::vector<Vector3f> convex_hull;
		int hull_size = buildConvexHull(extend_leaf_points, convex_hull, u_axis, v_axis);
		convex_hull.push_back(convex_hull[0]); //v[n] = v[0]. close the convex hull

		//test_sample_points_1 = extend_leaf_points;
		//test_sample_points_2 = convex_hull;
		//for (auto &p : test_sample_points_1)
		//	p[base_plane_norm_axis] = voxel_bmin[base_plane_norm_axis];
		//for (auto &p : test_sample_points_2)
		//	p[base_plane_norm_axis] = voxel_bmin[base_plane_norm_axis];

		for (int i = 0; i <= width; ++i) {
			for (int j = 0; j <= height; ++j) {
				Vector3f sampled_co;
				sampled_co[u_axis] = node_bmin_i[u_axis] + i * sampling_resolution_;
				sampled_co[v_axis] = node_bmin_i[v_axis] + j * sampling_resolution_;
				sampled_co[base_plane_norm_axis] = voxel_bmin[base_plane_norm_axis];

				if (!IsPointInsidePoly(sampled_co, convex_hull, convex_hull.size() - 1, u_axis, v_axis))
				{
					sampled_co[base_plane_norm_axis] = voxel_bmin[base_plane_norm_axis];
					continue;
				}

				radius_heights.clear(); radius_sqr_dsts.clear();
				bool inside_bounds = searchRadiusOnPlane(node_avg_norm, node_avg_center,
					sampled_co, sample_radius_search, u_axis, v_axis, base_plane_norm_axis,
					voxel_bmin, voxel_bmax, radius_heights, radius_sqr_dsts);

				if (!inside_bounds)
					continue;

				//interpolate
				float total_weight = 0.0f;
				float h = 0.0f;
				for (size_t k = 0; k < radius_heights.size(); ++k)
				{
					const float w = (radius_sqr_dsts[k] != 0.0f) ? (1.0f / sqrt(radius_sqr_dsts[k])) : 1.0f;
					const float h_tmp = radius_heights[k];
					//assert(h_tmp >= 0.0f);
					total_weight += w;
					h += h_tmp * w;
				}

				h = h / total_weight;
				if(h < 0.0f || h > voxel_height)
					continue;

				sampled_co[base_plane_norm_axis] = voxel_bmin[base_plane_norm_axis] + h;
				PointT sampled_p;
				sampled_p.x = sampled_co.x(); sampled_p.y = sampled_co.y(); sampled_p.z = sampled_co.z();

				output.points.push_back(sampled_p);
			}
		}
	}
}


template <typename PointT>
bool pcl::OctreeSampling<PointT>::searchRadiusOnPlane(
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

#if 1
	float intersect_height = search_p_3d[height_axis] - leaf_bmin[height_axis];
	if (intersect_height < 0 || intersect_height > (leaf_bmax[height_axis] - leaf_bmin[height_axis]))
		return false;
#endif

	PointT p; p.x = search_p_3d[0]; p.y = search_p_3d[1]; p.z = search_p_3d[2];
	
	std::vector<int> indices_3d;
	std::vector<float> dst_3d;
	int npoints = octree_->radiusSearch(p, radius, indices_3d, dst_3d);
	ret_heights.resize(npoints);
	ret_sqrt_dst.resize(npoints);

	Vector3f avg = Vector3f::Zero();
	for (int i = 0; i < npoints ; ++i)
	{
		const Vector3f tmp = this->input_->points[indices_3d[i]].getVector3fMap();
		ret_heights[i]  = tmp[height_axis] - leaf_bmin[height_axis];
		ret_sqrt_dst[i] = squareDistance(search_p_3d, tmp, u, v);

		avg += tmp;
	}

	test_sample_points_1.push_back(avg/npoints);

	return true;
}

template <typename PointT>
size_t pcl::OctreeSampling<PointT>::searchPointsRadius(const Eigen::Vector3f &center, float radius, std::vector<Eigen::Vector3f> *points, std::vector<float> *dsts)
{
	std::vector<int> indices_3d;
	std::vector<float> dst_3d;
	PointT p; p.x = center[0]; p.y = center[1]; p.z = center[2];
	int npoints = octree_->radiusSearch(p, radius, indices_3d, dst_3d);
	for (int i = 0; i < npoints; ++i)
		if (points)
			points->push_back(this->input_->points[indices_3d[i]].getVector3fMap());
	if (dsts)
		*dsts = std::move(dst_3d);

	return indices_3d.size();
}

template <typename PointT>
size_t pcl::OctreeSampling<PointT>::findBasePlane(const Eigen::Vector3f &avg_norm) const
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
void pcl::OctreeSampling<PointT>::averagePlane(const LeafNode *leaf, Eigen::Vector3f &p, Eigen::Vector3f &n) const
{
	const octree::OctreeContainerPointIndices *container = static_cast<const octree::OctreeContainerPointIndices*>(leaf->getContainerPtr());
	const std::vector<int>& indices = container->getPointIndicesVector();
	averagePlane(indices, p, n);
}

template <typename PointT>
void pcl::OctreeSampling<PointT>::averagePlane(const std::vector<int> &indices, Eigen::Vector3f &p, Eigen::Vector3f &n) const
{
	assert(input_normal_cloud_);

	Eigen::Vector3f avg_norm = Vector3f::Zero();
	Eigen::Vector3f avg = Vector3f::Zero();
	for (auto it = indices.cbegin(); it != indices.cend(); ++it)
	{
		const Eigen::Vector3f& norm = input_normal_cloud_->at(*it).getNormalVector3fMap();
		avg_norm += norm;
		avg += this->input_->points[*it].getVector3fMap();
	}
	avg_norm /= indices.size();
	avg_norm.normalize();

	p = avg / indices.size();
	n = avg_norm;
}

template <typename PointT>
void pcl::OctreeSampling<PointT>::calcBounds(const std::vector<int> &indices, Eigen::Vector3f &bmin, Eigen::Vector3f &bmax)
{
	typename std::vector<int>::const_iterator it;
	typename std::vector<int>::const_iterator it_end = indices.cend();

	bmin[0] = bmin[1] = bmin[2] =  std::numeric_limits <float>::max();
	bmax[0] = bmax[1] = bmax[2] = -std::numeric_limits <float>::max();

	for (it = indices.cbegin(); it != it_end; ++it)
	{
		const Eigen::Vector3f& p = this->input_->at(*it).getVector3fMap();
		bmin = bmin.array().min(p.array());
		bmax = bmax.array().max(p.array());
	}
}

template <typename PointT>
void pcl::OctreeSampling<PointT>::averagePoint(const std::vector<int> &indices, PointT &r_avg)
{
	typename std::vector<int>::const_iterator it;
	typename std::vector<int>::const_iterator it_end = indices.cend();
	Eigen::Vector3f avg = Vector3f::Zero();
	for (it = indices.cbegin(); it != it_end; ++it)
	{
		const Eigen::Vector3f& p = this->input_->at(*it).getVector3fMap();
		avg += p;
	}
	avg /= indices.size();

	r_avg.x = avg[0];
	r_avg.y = avg[1];
	r_avg.z = avg[2];
}

template <typename PointT>
void pcl::OctreeSampling<PointT>::heightApproximate(const std::vector<int> &indices, PointT &sample_p)
{
	Vector3f avg_p, avg_n, bmin, bmax;

	if (indices.size() == 0) {
		return;
	}else if (indices.size() == 1){
		const Vector3f &p = this->input_->points[indices[0]].getVector3fMap();
		sample_p.x = p.x();
		sample_p.y = p.y();
		sample_p.z = p.z();
	}

	averagePlane(indices, avg_p, avg_n);
	calcBounds(indices, bmin, bmax);
	size_t base_plane_axis = findBasePlane(avg_n);
	size_t u_axis = (base_plane_axis + 1) % 3;
	size_t v_axis = (base_plane_axis + 2) % 3;

	float h = 0.0f;
	float total_weight = 0.0f;
	for (size_t k = 0; k < indices.size(); ++k)
	{
		const Vector3f &p = this->input_->points[indices[k]].getVector3fMap();
		const float dst = squareDistance(avg_p, p, u_axis, v_axis);
		const float w = (dst != 0.0f) ? (1.0f / dst) : 1.0f;
		const float h_p = p[base_plane_axis] - bmin[base_plane_axis];
		total_weight += w;
		h += h_p * w;
	}
	h = h / total_weight;
	
	avg_p[base_plane_axis] = h + bmin[base_plane_axis];
	sample_p.x = avg_p.x();
	sample_p.y = avg_p.y();
	sample_p.z = avg_p.z();
}

template <typename PointT>
float pcl::OctreeSampling<PointT>::closestPoint(const std::vector<int> &indices, Eigen::Vector3f &center, PointT &closest)
{
	typename std::vector<int>::const_iterator it;
	typename std::vector<int>::const_iterator it_end = indices.cend();
	float min_sqr_dst = std::numeric_limits<float>::max();
	for (it = indices.cbegin(); it != it_end; ++it)
	{
		const Eigen::Vector3f& p = this->input_->at(*it).getVector3fMap();
		float sqr_dst = (p - center).squaredNorm();
		if (sqr_dst < min_sqr_dst)
		{
			min_sqr_dst = sqr_dst;
			closest = this->input_->at(*it);
		}
	}
	return  sqrt(min_sqr_dst);
}
#endif