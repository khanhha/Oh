/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2010-2012, Willow Garage, Inc.
 *
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 * $Id$
 *
 */

#include <pcl/pcl_config.h>

#ifndef PCL_SURFACE_IMPL_CONVEX_HULL_H_
#define PCL_SURFACE_IMPL_CONVEX_HULL_H_

#include <pcl/surface/convex_hull.h>
#include <pcl/common/common.h>
#include <pcl/common/eigen.h>
#include <pcl/common/transforms.h>
#include <pcl/common/io.h>
#include <stdio.h>
#include <stdlib.h>
 //#include <pcl/surface/qhull.h>


using namespace  Eigen;

//hull_center is zero
//hull vertices is already sorted in a circle around the center
template<typename PointInT>
float pcl::ConvexHull<PointInT>::computeConvexHull2DArea(const std::vector<std::pair<int, Eigen::Vector2f>> &hull) const
{
	float tot_area = 0.0f;
	size_t n = hull.size();
	for (size_t i = 0; i < n; ++i)
	{
		const auto &p0 = hull[i].second;
		const auto &p1 = hull[(i + 1) % n].second;
		float area = 0.5f * std::abs(p0[0] * p1[1] - p0[1] * p1[0]);
		tot_area += area;
	}
	return tot_area;
}

template<typename PointInT>
Vector3f pcl::ConvexHull<PointInT>::toVector3f(const qhull_point_type &point) const
{
	Vector3f p;
	assert(point.size() == 3);
	p[0] = point[0];
	p[1] = point[1];
	p[2] = point[2];
	return p;
}

template<typename PointInT>
void pcl::ConvexHull<PointInT>::computerVolumeArea(const qhull_type &hull, double &vol, double &area) const
{
	Vector3f centroid = Vector3f::Zero();
	size_t n = 0;
	for (auto &facet_ : hull.facets_)
	{
		for (auto &vertex_ : facet_.vertices_)
		{
			centroid += toVector3f(*vertex_);
			n++;
		}
	}

	centroid /= static_cast<float>(n);

	Vector3f trig[3];
	const float scale = 1.0f / 6.0f;
	double tot_area = 0.0f;
	double tot_vol = 0.0f;

	for (auto &facet_ : hull.facets_)
	{
		size_t i = 0;
		for (auto &vertex_ : facet_.vertices_)
			trig[i++] = toVector3f(*vertex_);

		Vector3f a = trig[1] - trig[0];
		Vector3f b = trig[2] - trig[0];
		Vector3f axb = a.cross(b);
		Vector3f c = centroid - trig[0];
		float area = 0.5f * axb.norm();
		float vol = scale * std::abs(axb.dot(c));
		tot_area += area;
		tot_vol += vol;
	}

	vol = tot_vol;
	area = tot_area;
}

//////////////////////////////////////////////////////////////////////////
template <typename PointInT> void
pcl::ConvexHull<PointInT>::calculateInputDimension()
{
	PCL_DEBUG("[pcl::%s::calculateInputDimension] WARNING: Input dimension not specified.  Automatically determining input dimension.\n", getClassName().c_str());
	Eigen::Vector4d xyz_centroid;
	compute3DCentroid(*input_, *indices_, xyz_centroid);
	EIGEN_ALIGN16 Eigen::Matrix3d covariance_matrix = Eigen::Matrix3d::Zero();
	computeCovarianceMatrixNormalized(*input_, *indices_, xyz_centroid, covariance_matrix);

	EIGEN_ALIGN16 Eigen::Vector3d eigen_values;
	pcl::eigen33(covariance_matrix, eigen_values);

	if (std::abs(eigen_values[0]) < std::numeric_limits<double>::epsilon() || std::abs(eigen_values[0] / eigen_values[2]) < 1.0e-3)
		dimension_ = 2;
	else
		dimension_ = 3;
}

//////////////////////////////////////////////////////////////////////////
template <typename PointInT> void
pcl::ConvexHull<PointInT>::performReconstruction2D(PointCloud &hull, std::vector<pcl::Vertices> &polygons, bool)
{
	qhull_value_type eps = std::numeric_limits<qhull_value_type>::epsilon();
	qhull_points_type points(indices_->size());
	qhull_type quick_hull_(2, eps);
	typename qhull_type::point_list initial_simplex_;

	int dimension = 2;
	bool xy_proj_safe = true;
	bool yz_proj_safe = true;
	bool xz_proj_safe = true;

	// Check the input's normal to see which projection to use
	PointInT p0 = input_->points[(*indices_)[0]];
	PointInT p1 = input_->points[(*indices_)[indices_->size() - 1]];
	PointInT p2 = input_->points[(*indices_)[indices_->size() / 2]];
	Eigen::Array4f dy1dy2 = (p1.getArray4fMap() - p0.getArray4fMap()) / (p2.getArray4fMap() - p0.getArray4fMap());
	while (!((dy1dy2[0] != dy1dy2[1]) || (dy1dy2[2] != dy1dy2[1])))
	{
		p0 = input_->points[(*indices_)[rand() % indices_->size()]];
		p1 = input_->points[(*indices_)[rand() % indices_->size()]];
		p2 = input_->points[(*indices_)[rand() % indices_->size()]];
		dy1dy2 = (p1.getArray4fMap() - p0.getArray4fMap()) / (p2.getArray4fMap() - p0.getArray4fMap());
	}

	pcl::PointCloud<PointInT> normal_calc_cloud;
	normal_calc_cloud.points.resize(3);
	normal_calc_cloud.points[0] = p0;
	normal_calc_cloud.points[1] = p1;
	normal_calc_cloud.points[2] = p2;

	Eigen::Vector4d normal_calc_centroid;
	Eigen::Matrix3d normal_calc_covariance;
	pcl::compute3DCentroid(normal_calc_cloud, normal_calc_centroid);
	pcl::computeCovarianceMatrixNormalized(normal_calc_cloud, normal_calc_centroid, normal_calc_covariance);

	// Need to set -1 here. See eigen33 for explanations.
	Eigen::Vector3d::Scalar eigen_value;
	Eigen::Vector3d plane_params;
	pcl::eigen33(normal_calc_covariance, eigen_value, plane_params);
	float theta_x = fabsf(static_cast<float> (plane_params.dot(x_axis_)));
	float theta_y = fabsf(static_cast<float> (plane_params.dot(y_axis_)));
	float theta_z = fabsf(static_cast<float> (plane_params.dot(z_axis_)));

	// Check for degenerate cases of each projection
	// We must avoid projections in which the plane projects as a line
	if (theta_z > projection_angle_thresh_)
	{
		xz_proj_safe = false;
		yz_proj_safe = false;
	}
	if (theta_x > projection_angle_thresh_)
	{
		xz_proj_safe = false;
		xy_proj_safe = false;
	}
	if (theta_y > projection_angle_thresh_)
	{
		xy_proj_safe = false;
		yz_proj_safe = false;
	}


	// Build input data, using appropriate projection
	int j = 0;
	if (xy_proj_safe)
	{
		for (size_t i = 0; i < indices_->size(); ++i)
		{
			points[i].resize(2);
			points[i][0] = static_cast<float> (input_->points[(*indices_)[i]].x);
			points[i][1] = static_cast<float> (input_->points[(*indices_)[i]].y);
		}
	}
	else if (yz_proj_safe)
	{
		for (size_t i = 0; i < indices_->size(); ++i, j += dimension)
		{
			points[i].resize(2);
			points[i][0] = static_cast<float> (input_->points[(*indices_)[i]].y);
			points[i][1] = static_cast<float> (input_->points[(*indices_)[i]].z);
		}
	}
	else if (xz_proj_safe)
	{
		for (size_t i = 0; i < indices_->size(); ++i, j += dimension)
		{
			points[i].resize(2);
			points[i][0] = static_cast<float> (input_->points[(*indices_)[i]].x);
			points[i][1] = static_cast<float> (input_->points[(*indices_)[i]].z);
		}
	}
	else
	{
		// This should only happen if we had invalid input
		PCL_ERROR("[pcl::%s::performReconstruction2D] Invalid input!\n", getClassName().c_str());
	}

	quick_hull_.add_points(points.begin(), points.end());
	initial_simplex_ = quick_hull_.get_affine_basis();

	quick_hull_.create_initial_simplex(std::cbegin(initial_simplex_), std::prev(std::cend(initial_simplex_)));

	quick_hull_.create_convex_hull();

	//if (!quick_hull_.check()) {
	//	  std::cerr << "error: convex hull algorithm: resulting structure is not valid convex polytope" << std::endl;
	//	  return;
	//}


	std::vector<size_t> hull_vert_indices;

	for (auto const & facet_ : quick_hull_.facets_) {
		auto const & vertices_ = facet_.vertices_;
		hull_vert_indices.push_back(std::distance(points.begin(), vertices_.front()));
#if 0
		if (!facet_.coplanar_.empty()) {
			for (auto const & v : facet_.coplanar_) {
				for (value_type const & coordinate_ : *v) {
					std::cout << coordinate_ << ' ';
				}
				std::cout << '\n';
			}
			std::cout << "e\n";
		}
#endif
	}

	// Compute convex hull

	// 0 if no error from convex hull or it doesn't find any vertices
	if (quick_hull_.facets_.size() == 0)
	{
		PCL_ERROR("[pcl::%s::performReconstrution2D] ERROR: qhull was unable to compute a convex hull for the given point cloud (%lu)!\n", getClassName().c_str(), indices_->size());

		hull.points.resize(0);
		hull.width = hull.height = 0;
		polygons.resize(0);
		return;
	}

	int num_vertices = hull_vert_indices.size();
	hull.points.resize(num_vertices);
	memset(&hull.points[0], static_cast<int> (hull.points.size()), sizeof(PointInT));

	std::vector<std::pair<int, Eigen::Vector2f>> idx_points(num_vertices);
	idx_points.resize(hull.points.size());
	memset(&idx_points[0], static_cast<int> (hull.points.size()), sizeof(std::pair<int, Eigen::Vector4f>));

	for (size_t i = 0; i < hull_vert_indices.size(); ++i)
	{
		hull.points[i] = input_->points[(*indices_)[hull_vert_indices[i]]];
		idx_points[i].first = hull_vert_indices[i];
	}

	// Sort
	Eigen::Vector4f centroid;
	pcl::compute3DCentroid(hull, centroid);
	if (xy_proj_safe)
	{
		for (size_t j = 0; j < hull.points.size(); j++)
		{
			idx_points[j].second[0] = hull.points[j].x - centroid[0];
			idx_points[j].second[1] = hull.points[j].y - centroid[1];
		}
	}
	else if (yz_proj_safe)
	{
		for (size_t j = 0; j < hull.points.size(); j++)
		{
			idx_points[j].second[0] = hull.points[j].y - centroid[1];
			idx_points[j].second[1] = hull.points[j].z - centroid[2];
		}
	}
	else if (xz_proj_safe)
	{
		for (size_t j = 0; j < hull.points.size(); j++)
		{
			idx_points[j].second[0] = hull.points[j].x - centroid[0];
			idx_points[j].second[1] = hull.points[j].z - centroid[2];
		}
	}

	std::sort(idx_points.begin(), idx_points.end(), comparePoints2D);

	if (compute_area_)
	{
		total_area_ = computeConvexHull2DArea(idx_points);
		total_volume_ = 0.0;

		double sqr_cir_radius = 0.0f;
		for (auto & pair_ : idx_points)
		{
			float  tmp = pair_.second.squaredNorm();
			if (sqr_cir_radius < tmp)
				sqr_cir_radius = tmp;
		}
		perimeter_ = 2 * M_PI * sqrt(sqr_cir_radius);
	}

	polygons.resize(1);
	polygons[0].vertices.resize(hull.points.size());

	hull_indices_.header = input_->header;
	hull_indices_.indices.clear();
	hull_indices_.indices.reserve(hull.points.size());

	for (int j = 0; j < static_cast<int> (hull.points.size()); j++)
	{
		hull_indices_.indices.push_back((*indices_)[idx_points[j].first]);
		hull.points[j] = input_->points[(*indices_)[idx_points[j].first]];
		polygons[0].vertices[j] = static_cast<unsigned int> (j);
	}

	hull.width = static_cast<uint32_t> (hull.points.size());
	hull.height = 1;
	hull.is_dense = false;

	return;
}

#ifdef __GNUC__
#pragma GCC diagnostic ignored "-Wold-style-cast"
#endif
//////////////////////////////////////////////////////////////////////////
template <typename PointInT> void
pcl::ConvexHull<PointInT>::performReconstruction3D(
	PointCloud &hull, std::vector<pcl::Vertices> &polygons, bool fill_polygon_data)
{
	qhull_value_type const eps = std::numeric_limits<qhull_value_type>::epsilon();
	int dimension = 3;

	qhull_points_type points(indices_->size());
	qhull_type quick_hull_(3, eps);
	typename qhull_type::point_list initial_simplex_;

	for (size_t i = 0; i < indices_->size(); ++i)
	{
		points[i].resize(3);
		points[i][0] = static_cast<qhull_value_type>(input_->points[(*indices_)[i]].x);
		points[i][1] = static_cast<qhull_value_type>(input_->points[(*indices_)[i]].y);
		points[i][2] = static_cast<qhull_value_type>(input_->points[(*indices_)[i]].z);
	}

	quick_hull_.add_points(points.begin(), points.end());
	initial_simplex_ = quick_hull_.get_affine_basis();

	quick_hull_.create_initial_simplex(std::cbegin(initial_simplex_), std::prev(std::cend(initial_simplex_)));

	quick_hull_.create_convex_hull();

	if (quick_hull_.facets_.size() == 0)
	{
		PCL_ERROR("[pcl::%s::performReconstrution3D] ERROR: qhull was unable to compute a convex hull for the given point cloud (%lu)!\n", getClassName().c_str(), input_->points.size());

		hull.points.resize(0);
		hull.width = hull.height = 0;
		polygons.resize(0);
		return;
	}

	std::set<size_t> hull_vert_indices;

	for (auto const & facet_ : quick_hull_.facets_) {
		for (auto const & vertex_ : facet_.vertices_) {
			auto idx = std::distance(points.begin(), vertex_);
			hull_vert_indices.insert(idx);
		}

#if 0
		if (!facet_.coplanar_.empty()) {
			for (auto const & v : facet_.coplanar_) {
				for (value_type const & coordinate_ : *v) {
					std::cout << coordinate_ << ' ';
				}
			}
		}
#endif
	}

	for (auto idx : hull_vert_indices)
		hull.points.push_back(input_->points[idx]);

	int num_facets = quick_hull_.facets_.size();
	int num_vertices = hull_vert_indices.size();
	hull.points.resize(num_vertices);

	unsigned int max_vertex_id = 0;

	for (auto &idx : hull_vert_indices)
	{
		if (idx > max_vertex_id)
			max_vertex_id = idx;
	}

	++max_vertex_id;
	std::vector<int> qhid_to_pcidx(max_vertex_id);

	hull_indices_.header = input_->header;
	hull_indices_.indices.clear();
	hull_indices_.indices.reserve(num_vertices);

	size_t i = 0;
	for (auto &idx : hull_vert_indices)
	{
		// Add vertices to hull point_cloud and store index
		hull_indices_.indices.push_back((*indices_)[idx]);
		hull.points[i] = input_->points[(*indices_)[hull_indices_.indices.back()]];

		qhid_to_pcidx[idx] = i; // map the vertex id of qhull to the point cloud index
		++i;
	}

	if (compute_area_)
	{
		computerVolumeArea(quick_hull_, total_volume_, total_area_);
	}

	if (fill_polygon_data)
	{
		polygons.resize(num_facets);
		size_t dd = 0;

		for (auto const & facet_ : quick_hull_.facets_) {
			auto const & vertices_ = facet_.vertices_;
			polygons[dd].vertices.resize(3);

			size_t vertex_i = 0;
			for (auto const & vertex_ : facet_.vertices_) {
				polygons[dd].vertices[vertex_i++] = qhid_to_pcidx[std::distance(points.begin(), vertex_)];
			}
			++dd;
		}
	}

	hull.width = static_cast<uint32_t> (hull.points.size());
	hull.height = 1;
	hull.is_dense = false;
}

#ifdef __GNUC__
#pragma GCC diagnostic warning "-Wold-style-cast"
#endif

//////////////////////////////////////////////////////////////////////////
template <typename PointInT> void
pcl::ConvexHull<PointInT>::performReconstruction(PointCloud &hull, std::vector<pcl::Vertices> &polygons,
	bool fill_polygon_data)
{
	if (dimension_ == 0)
		calculateInputDimension();
	if (dimension_ == 2)
		performReconstruction2D(hull, polygons, fill_polygon_data);
	else if (dimension_ == 3)
		performReconstruction3D(hull, polygons, fill_polygon_data);
	else
		PCL_ERROR("[pcl::%s::performReconstruction] Error: invalid input dimension requested: %d\n", getClassName().c_str(), dimension_);
}

//////////////////////////////////////////////////////////////////////////
template <typename PointInT> void
pcl::ConvexHull<PointInT>::reconstruct(PointCloud &points)
{
	points.header = input_->header;
	if (!initCompute() || input_->points.empty() || indices_->empty())
	{
		points.points.clear();
		return;
	}

	// Perform the actual surface reconstruction
	std::vector<pcl::Vertices> polygons;
	performReconstruction(points, polygons, false);

	points.width = static_cast<uint32_t> (points.points.size());
	points.height = 1;
	points.is_dense = true;

	deinitCompute();
}


//////////////////////////////////////////////////////////////////////////
template <typename PointInT> void
pcl::ConvexHull<PointInT>::performReconstruction(PolygonMesh &output)
{
	// Perform reconstruction
	pcl::PointCloud<PointInT> hull_points;
	performReconstruction(hull_points, output.polygons, true);

	// Convert the PointCloud into a PCLPointCloud2
	pcl::toPCLPointCloud2(hull_points, output.cloud);
}

//////////////////////////////////////////////////////////////////////////
template <typename PointInT> void
pcl::ConvexHull<PointInT>::performReconstruction(std::vector<pcl::Vertices> &polygons)
{
	pcl::PointCloud<PointInT> hull_points;
	performReconstruction(hull_points, polygons, true);
}

//////////////////////////////////////////////////////////////////////////
template <typename PointInT> void
pcl::ConvexHull<PointInT>::reconstruct(PointCloud &points, std::vector<pcl::Vertices> &polygons)
{
	points.header = input_->header;
	if (!initCompute() || input_->points.empty() || indices_->empty())
	{
		points.points.clear();
		return;
	}

	// Perform the actual surface reconstruction
	performReconstruction(points, polygons, true);

	points.width = static_cast<uint32_t> (points.points.size());
	points.height = 1;
	points.is_dense = true;

	deinitCompute();
}
//////////////////////////////////////////////////////////////////////////
template <typename PointInT> void
pcl::ConvexHull<PointInT>::getHullPointIndices(pcl::PointIndices &hull_point_indices) const
{
	hull_point_indices = hull_indices_;
}

#define PCL_INSTANTIATE_ConvexHull(T) template class PCL_EXPORTS pcl::ConvexHull<T>;
#endif    // PCL_SURFACE_IMPL_CONVEX_HULL_H_
