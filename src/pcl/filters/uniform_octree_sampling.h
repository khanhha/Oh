/*
* Software License Agreement (BSD License)
*
*  Point Cloud Library (PCL) - www.pointclouds.org
*  Copyright (c) 2010-2011, Willow Garage, Inc.
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

#ifndef PCL_FILTERS_UNIFORM_OCTREE_SAMPLING_H_
#define PCL_FILTERS_UNIFORM_OCTREE_SAMPLING_H_

#include <memory>
#include <vector>
#include <unordered_map>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/filter.h>
#include <pcl/octree/octree.h>

namespace pcl
{
	/** \brief @b UniformSampling assembles a local 3D grid over a given PointCloud, and downsamples + filters the data.
	*
	* The @b UniformSampling class creates a *3D voxel grid* (think about a voxel
	* grid as a set of tiny 3D boxes in space) over the input point cloud data.
	* Then, in each *voxel* (i.e., 3D box), all the points present will be
	* approximated (i.e., *downsampled*) with their centroid. This approach is
	* a bit slower than approximating them with the center of the voxel, but it
	* represents the underlying surface more accurately.
	*
	* \author Radu Bogdan Rusu
	* \ingroup keypoints
	*/
	template <typename PointT>
	class UniformOctreeSampling : public Filter<PointT>
	{
		using Filter<PointT>::filter_name_;
		using Filter<PointT>::input_;
		using Filter<PointT>::indices_;
		using Filter<PointT>::getClassName;

		typedef pcl::octree::OctreePointCloudNormal<PointT, Normal> OctreeNormal;
		typedef typename OctreeNormal::Ptr OctreeNormalPtr;
		typedef typename OctreeNormal::LeafNode LeafNode;
	public:
		typedef std::shared_ptr<UniformOctreeSampling<PointT> > Ptr;
		typedef std::shared_ptr<const UniformOctreeSampling<PointT> > ConstPtr;

		typedef pcl::PointCloud<Normal>			NormalCloud;
		typedef typename NormalCloud::Ptr		NormalCloudPtr;
		typedef typename NormalCloud::ConstPtr	NormalCloudconstPtr;
		typedef typename Filter<PointT>::PointCloud PointCloud;


		/** \brief Empty constructor. */
		UniformOctreeSampling() :
			sampling_size_(Eigen::Vector3f::Zero()),
			inverse_sampling_size_(Eigen::Vector3f::Zero()),
			min_b_(Eigen::Vector3f::Zero()),
			max_b_(Eigen::Vector3f::Zero()),
			min_bi_(Eigen::Vector3i::Zero()),
			max_bi_(Eigen::Vector3i::Zero())
		{
			filter_name_ = "UniformOctreeSampling";
		}

		/** \brief Destructor. */
		virtual ~UniformOctreeSampling()
		{
		}

		inline void setInputNormalCloud(NormalCloudconstPtr nm) { input_normal_cloud_ = nm; }
		inline void setOctreeResolution(double rel) { octree_resolution_ = rel; }
		inline void setOctreeNormalThreshold(double thres) { octree_normal_threshold = thres; }
		inline void setSamplingResolution(double rel) 
		{
			sampling_resolution_ = rel; 
			sampling_size_[0] = sampling_size_[1] = sampling_size_[2] = static_cast<float> (sampling_resolution_);
			inverse_sampling_size_ = Eigen::Array3f::Ones() / sampling_size_.array();
		}

		inline void setSampleRadiusSearch(double radius) { sample_radius_search = radius; };

	public:
		std::vector<Eigen::Vector3f> test_sample_points;
		std::vector<Eigen::Vector3f> test_sample_points_1;
		std::vector<Eigen::Vector3f> test_node_points;
		std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> test_node_bounds;
	protected:
		/** \brief Simple structure to hold an nD centroid and the number of points in a leaf. */
		NormalCloudconstPtr input_normal_cloud_;
		double octree_normal_threshold;
		double octree_resolution_;
		double sampling_resolution_;
		double sample_radius_search;
		
		Eigen::Vector3f  sampling_size_;
		Eigen::Vector3f  inverse_sampling_size_;
		
		/** \brief The minimum and maximum bin coordinates, the number of divisions, and the division multiplier. */
		Eigen::Vector3f min_b_, max_b_;
		Eigen::Vector3i min_bi_, max_bi_;
		OctreeNormalPtr _octree;
		/** \brief Downsample a Point Cloud using a voxelized grid approach
		* \param[out] output the resultant point cloud message
		*/
		void
			applyFilter(PointCloud &output);
		size_t 
			findBasePlane(const LeafNode *node) const;
		size_t 
			searchRadiusOnPlane(const std::vector<int> &all_indices, const Eigen::Vector3f &search_p, float sqr_radius, size_t u, size_t v, std::vector<int> &ret_indices, std::vector<float> &ret_sqrt_dst) const;
		inline float
			heightBasePlane(const size_t &idx, const size_t &axis, const Eigen::Vector3f &bmin)
		{
			if (axis == 0)
				return input_->points[idx].x - bmin.x();
			else if (axis == 1)
				return input_->points[idx].y - bmin.y();
			else
				return input_->points[idx].z - bmin.z();
		}

		inline float interpolationWeight(const size_t &idx, const Eigen::Vector3f &p, const size_t &u, const size_t &v)
		{
			const Eigen::Vector3f &other_p = input_->points[idx].getVector3fMap();
			const float du = (other_p[u] - p[u]);
			const float dv = (other_p[v] - p[v]);
			const float sqr_dst = du * du + dv*dv;
			if (sqr_dst > 0)
				return 1.0f / sqrt(sqr_dst);
			else
				return 1.0f;
		}

		void calcBounds(const std::vector<int> &indices, Eigen::Vector3f &bmin, Eigen::Vector3f &bmax);

		inline void
			initSamplingBounds(const Eigen::Vector3f &min_p, const Eigen::Vector3f &max_p)
		{
			min_b_ = min_p;
			max_b_ = max_p;
			// Compute the minimum and maximum bounding box values
			for (size_t i = 0; i < 3; ++i) {
				min_b_[i] = static_cast<int> (round(min_p[i] * inverse_sampling_size_[i]));
				max_b_[i] = static_cast<int> (round(max_p[i] * inverse_sampling_size_[i]));
			}
		};

		inline Eigen::Vector3i
			mapSamplingCoord(const Eigen::Vector3f &in) const
		{
			Eigen::Vector3i ijk = Eigen::Vector3i::Zero();
			ijk[0] = static_cast<int> (round(in[0] * inverse_sampling_size_[0]));
			ijk[1] = static_cast<int> (round(in[1] * inverse_sampling_size_[1]));
			ijk[2] = static_cast<int> (round(in[2] * inverse_sampling_size_[2]));
			return ijk;
		};
	};
}

#ifdef PCL_NO_PRECOMPILE
#include <pcl/filters/impl/uniform_octree_sampling.hpp>
#endif

#endif  //#ifndef PCL_FILTERS_UNIFORM_SAMPLING_H_

