/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2010-2011, Willow Garage, Inc.
 *  Copyright (c) 2012-, Open Perception, Inc.
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
 *   * Neither the name of the copyright holder(s) nor the names of its
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
 * $Id: octree_pointcloud_voxelcentroid.hpp 6459 2012-07-18 07:50:37Z dpb $
 */

#ifndef PCL_OCTREE_NORMAL_HPP
#define PCL_OCTREE_NORMAL_HPP

#define OH_DEBUG 1
#ifdef OH_DEBUG
#include <Windows.h>
#endif
#include <pcl/octree/impl/octree_pointcloud.hpp>

template<typename PointT, typename NormalT, typename LeafContainerT /*= OctreeContainerPointIndices*/, typename BranchContainerT /*= OctreeContainerEmpty */>
void pcl::octree::OctreePointCloudNormal<PointT, NormalT, LeafContainerT, BranchContainerT>::addPointsFromInputCloud()
{
	pcl::octree::OctreePointCloud<PointT, LeafContainerT, BranchContainerT>::addPointsFromInputCloud();
	
	if (!use_normal_threshold_) 
	{
		return;
	}
	else
	{
		std::vector<OctreeKey> expand_leaf_keys;



		OctreeT::LeafNodeIterator leafIter, leafEnd = leaf_end();
		for (leafIter = leaf_begin(); leafIter != leafEnd; ++leafIter)
		{
			LeafNode *leaf = static_cast<LeafNode*>(*leafIter);
			OctreeContainerPointIndices *container = static_cast<OctreeContainerPointIndices*>(leaf->getContainerPtr());

			if (container && container->getSize() > 0)
			{
				float normal_deviation = normalThreshold(container);
				if (normal_deviation < leaf_normal_threshold_)
				{
					const PointT &point_sample = input_->points[container->getPointIndicesVector()[0]];
					OctreeKey    point_key;
					genOctreeKeyforPoint(point_sample, point_key);
					expand_leaf_keys.push_back(point_key);
				}
			}
		}

		for (auto it = expand_leaf_keys.begin(); it != expand_leaf_keys.end(); ++it)
		{
			const OctreeKey& leaf_key = *it;

			std::stack<OctreeKey> splitLeafs;
			splitLeafs.push(leaf_key);

			while (!splitLeafs.empty())
			{
				LeafNode *leaf = nullptr; BranchNode *branch = nullptr;

				OctreeKey s = splitLeafs.top();
				splitLeafs.pop();

				unsigned int  depth_search = findLeafRecursive(s, this->depth_mask_, this->root_node_, leaf, branch);
				if (leaf == nullptr)
				{
					cerr << "no leaf found at key " << s.x << ", " << s.y << ", " << s.z << std::endl;
				}
				else
				{
					if (depth_search == 0)
					{
						cerr << "couldn't split more. reach the limit of octree resolution";
					}
					else
					{
						unsigned char child_idx = s.getChildIdxWithDepthMask(depth_search << 1);

						assert(getBranchChildPtr(*branch, child_idx)->getNodeType() == LEAF_NODE);
						expandLeafNode(leaf, branch, child_idx, depth_search);
						assert(getBranchChildPtr(*branch, child_idx)->getNodeType() == BRANCH_NODE); //the leaf node is now replaced by a  new branch node

						BranchNode *new_branch = static_cast<BranchNode*>(getBranchChildPtr(*branch, child_idx));
						assert(new_branch != nullptr);

						for (unsigned char i = 0; i < 8; ++i)
						{
							if (branchHasChild(*new_branch, i) && getBranchChildPtr(*new_branch, i)->getNodeType() == LEAF_NODE)
							{
								LeafNode *new_leaf = static_cast<LeafNode*>(getBranchChildPtr(*new_branch, i));
								OctreeContainerPointIndices *container = static_cast<OctreeContainerPointIndices*>(new_leaf->getContainerPtr());

								if (container->getSize() > 1 && normalThreshold(container) < leaf_normal_threshold_)
								{
									const PointT &point_sample = input_->points[container->getPointIndicesVector()[0]];
									OctreeKey    point_key;
									genOctreeKeyforPoint(point_sample, point_key);
									splitLeafs.push(point_key);
								}
							}
						}
					}
				}
			}
		}
	}
}


template<typename PointT, typename NormalT, typename LeafContainerT /*= OctreeContainerPointIndices*/, typename BranchContainerT /*= OctreeContainerEmpty */>
float pcl::octree::OctreePointCloudNormal<PointT, NormalT, LeafContainerT, BranchContainerT>::leafSize(OctreeContainerPointIndices *container)
{
	const std::vector<int>& indices = container->getPointIndicesVector();
	typename std::vector<int>::const_iterator it;
	typename std::vector<int>::const_iterator it_end = indices.cend();
	Eigen::Vector3f bmin, bmax;

	bmin[0] = bmin[1] = bmin[2] =  std::numeric_limits <float>::max();
	bmax[0] = bmax[1] = bmax[2] = -std::numeric_limits <float>::max();

	for (it = indices.cbegin(); it != it_end; ++it)
	{
		const Eigen::Vector3f& p = input_->at(*it).getVector3fMap();
		bmin = bmin.array().min(p.array());
		bmax = bmax.array().max(p.array());
	}

	Eigen::Vector3f d;
	d[0] = bmax[0] - bmin[0];
	d[1] = bmax[1] - bmin[1];
	d[2] = bmax[2] - bmin[2];
	return d.maxCoeff();
}

template<typename PointT, typename NormalT, typename LeafContainerT /*= OctreeContainerPointIndices*/, typename BranchContainerT /*= OctreeContainerEmpty */>
float pcl::octree::OctreePointCloudNormal<PointT, NormalT, LeafContainerT, BranchContainerT>::normalThreshold(OctreeContainerPointIndices *container)
{
	const std::vector<int>& indices = container->getPointIndicesVector();
	typename std::vector<int>::const_iterator it;
	typename std::vector<int>::const_iterator it_end = indices.cend();
	
	Eigen::Vector3f avg_norm = Eigen::Vector3f::Zero();
	for (it = indices.cbegin(); it != it_end; ++it)
	{
		const Eigen::Vector3f& norm = normal_input_->at(*it).getNormalVector3fMap();
		avg_norm += norm;
	}
	avg_norm /= indices.size();
	avg_norm.normalize();

	float dot_avg = 0.0f;
	for (it = indices.cbegin(); it != it_end; ++it)
	{
		const Eigen::Vector3f& norm = normal_input_->at(*it).getNormalVector3fMap();
		dot_avg += norm.dot(avg_norm);
	}
	dot_avg /= indices.size();

	return dot_avg;
}

#define PCL_INSTANTIATE_OctreePointCloudNormal(T) template class PCL_EXPORTS pcl::octree::OctreePointCloudNormal<T>;

#endif

