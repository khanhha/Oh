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
*/

#ifndef PCL_OCTREE_NORMAL_H_
#define PCL_OCTREE_NORMAL_H_

#include <memory>
#include <pcl/point_cloud.h>
#include <pcl/octree/octree_search.h>

namespace pcl
{
	namespace octree
	{
		template<typename PointT, typename NormalT, typename LeafContainerT = OctreeContainerPointIndices, typename BranchContainerT = OctreeContainerEmpty >
		class OctreePointCloudNormal: public OctreePointCloudSearch<PointT, LeafContainerT, BranchContainerT>
		{
		public:
			// public typedefs
			typedef pcl::PointCloud<NormalT> PointCloudNormal;
			typedef std::shared_ptr<PointCloudNormal> PointCloudNormalPtr;
			typedef std::shared_ptr<const PointCloudNormal> PointCloudNormalConstPtr;

			typedef std::shared_ptr<std::vector<int> > IndicesPtr;
			typedef std::shared_ptr<const std::vector<int> > IndicesConstPtr;

			// Boost shared pointers
			typedef std::shared_ptr<OctreePointCloudNormal<PointT, NormalT, LeafContainerT, BranchContainerT> > Ptr;
			typedef std::shared_ptr<const OctreePointCloudNormal<PointT, NormalT, LeafContainerT, BranchContainerT> > ConstPtr;

			typedef OctreePointCloud<PointT, LeafContainerT, BranchContainerT> OctreeT;
			typedef typename OctreeT::LeafNode LeafNode;
			typedef typename OctreeT::BranchNode BranchNode;

			/** \brief Constructor.
			* \param[in] resolution octree resolution at lowest octree level
			*/
			OctreePointCloudNormal(const double resolution) :
				OctreePointCloudSearch<PointT, LeafContainerT, BranchContainerT>(resolution),
				use_normal_threshold_(false),
				leaf_normal_threshold_(0.0f)
			{
			}

			/** \brief Empty class destructor. */
			virtual ~OctreePointCloudNormal()
			{
			}
			/** \brief Provide a pointer to the input data set.
			* \param[in] cloud_arg the const boost shared pointer to a PointCloud message
			* \param[in] indices_arg the point indices subset that is to be used from \a cloud - if 0 the whole point cloud is used
			*/
			inline void setInputNormalCloud(const PointCloudNormalConstPtr &cloud_arg, const IndicesConstPtr &indices_arg = IndicesConstPtr())
			{
				normal_input_ = cloud_arg;
				normal_indices_ = indices_arg;
			}

			
			/** \brief any leaf node with the normal mean deviation smaller than this value will be split until the octree resolution is reached
			*/
			inline void setNormalThreshold(float value)
			{
				assert(value > 0.0f && value < 1.0f);
				leaf_normal_threshold_ = value;
				use_normal_threshold_ = true;
			}

			void addPointsFromInputCloud();

		public:
			float  leafSize(OctreeContainerPointIndices *container);
			float  normalThreshold(OctreeContainerPointIndices *container);

			PointCloudNormalConstPtr normal_input_;
			IndicesConstPtr normal_indices_;
			float			leaf_normal_threshold_;
			bool			use_normal_threshold_;
		};

	}
}

//#ifdef PCL_NO_PRECOMPILE
#include <pcl/octree/impl/octree_pointcloud_normal.hpp>
//#endif

#endif    // PCL_OCTREE_SEARCH_H_
