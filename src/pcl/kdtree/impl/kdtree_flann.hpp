/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2009-2011, Willow Garage, Inc.
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
 */

#ifndef PCL_KDTREE_KDTREE_IMPL_FLANN_H_
#define PCL_KDTREE_KDTREE_IMPL_FLANN_H_

#include <cstdio>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/console/print.h>

///////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT, typename Dist>
pcl::KdTreeFLANN<PointT, Dist>::KdTreeFLANN (bool sorted)
  : pcl::KdTree<PointT> (sorted)
  , data_adaptor_(new DataAdaptor())
  , flann_index_ ()
  , dim_ (0), total_nr_points_ (0), max_points_per_leaf_(30)
  , param_k_ (nanoflann::SearchParams (-1 , epsilon_))
  , param_radius_ (nanoflann::SearchParams (-1, epsilon_, sorted))
{
	cache_radius_search_results.reserve(100);
}

///////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT, typename Dist>
pcl::KdTreeFLANN<PointT, Dist>::KdTreeFLANN (const KdTreeFLANN<PointT> &k) 
  : pcl::KdTree<PointT> (false)
  , flann_index_ ()
  , dim_ (0), total_nr_points_ (0), max_points_per_leaf_(30)
  , param_k_ (nanoflann::SearchParams (-1 , epsilon_))
  , param_radius_ (nanoflann::SearchParams (-1, epsilon_, false))
{
  *this = k;
  cache_radius_search_results.reserve(100);
}

///////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT, typename Dist> void 
pcl::KdTreeFLANN<PointT, Dist>::setEpsilon (float eps)
{
  epsilon_ = eps;
  param_k_ =  nanoflann::SearchParams (-1 , epsilon_);
  param_radius_ = nanoflann::SearchParams (-1 , epsilon_, sorted_);
}

///////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT, typename Dist> void 
pcl::KdTreeFLANN<PointT, Dist>::setSortedResults (bool sorted)
{
  sorted_ = sorted;
  param_k_ = nanoflann::SearchParams (-1, epsilon_);
  param_radius_ = nanoflann::SearchParams (-1, epsilon_, sorted_);
}

///////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT, typename Dist> void 
pcl::KdTreeFLANN<PointT, Dist>::setInputCloud (const PointCloudConstPtr &cloud, const IndicesConstPtr &indices)
{
  cleanup ();   // Perform an automatic cleanup of structures

  epsilon_ = 0.0f;   // default error bound value
  dim_ = point_representation_->getNumberOfDimensions (); // Number of dimensions - default is 3 = xyz

  input_   = cloud;
  indices_ = indices;

  data_adaptor_->set_point_cloud(cloud, indices);
  
  // Allocate enough data
  if (!input_)
  {
    PCL_ERROR ("[pcl::KdTreeFLANN::setInputCloud] Invalid input!\n");
    return;
  }

  if (indices != NULL)
  {
	  total_nr_points_ = indices->size();
  }
  else
  {
	  total_nr_points_ = input_->size();
  }

  if (total_nr_points_ == 0)
  {
    PCL_ERROR ("[pcl::KdTreeFLANN::setInputCloud] Cannot create a KDTree with an empty input cloud!\n");
    return;
  }


#if 0
  flann_index_.reset (new FLANNIndex (::flann::Matrix<float> (cloud_.get (), 
                                                              index_mapping_.size (), 
                                                              dim_),
                                      ::flann::KDTreeSingleIndexParams (15))); // max 15 points/leaf
  flann_index_->buildIndex ();
#endif
}

template <typename PointT, typename Dist /*= nanoflann::L2_Simple_Adaptor<float, PointCloudNanoflannAdaptor<PointT>>*/>
void pcl::KdTreeFLANN<PointT, Dist>::setNormalCloud(const NormalCloudConstPtr &cloud)
{
	data_adaptor_->set_normal_cloud(cloud);
}

template <typename PointT, typename Dist /*= nanoflann::L2_Simple_Adaptor<float, PointCloudNanoflannAdaptor<PointT>>*/>
void pcl::KdTreeFLANN<PointT, Dist>::addPointsFromInputCloud()
{
	flann_index_.reset(new FLANNIndex(dim_, *data_adaptor_, nanoflann::KDTreeSingleIndexNormalAdaptorParams(max_points_per_leaf_ /* max leaf */)));
	flann_index_->buildIndex();
}

///////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT, typename Dist> int 
pcl::KdTreeFLANN<PointT, Dist>::nearestKSearch (const PointT &point, int k, 
                                                std::vector<int> &k_indices, 
                                                std::vector<float> &k_distances) const
{
  assert (point_representation_->isValid (point) && "Invalid (NaN, Inf) point coordinates given to nearestKSearch!");

  if (k > total_nr_points_)
    k = total_nr_points_;

  k_indices.resize (k);
  k_distances.resize (k);
  
  std::vector<float> query (dim_);
  point_representation_->vectorize (static_cast<PointT> (point), query);

  // Wrap the k_indices and k_distances vectors (no data copy)
  flann_index_->knnSearch (&query[0], k, &k_indices[0], &k_distances[0]);

  // Do mapping to original point cloud
  if (indices_) 
  {
    for (size_t i = 0; i < static_cast<size_t> (k); ++i)
    {
      int& neighbor_index = k_indices[i];
      neighbor_index = (*indices_)[neighbor_index];
    }
  }

  return (k);
}

///////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT, typename Dist> int 
pcl::KdTreeFLANN<PointT, Dist>::radiusSearch (const PointT &point, double radius, std::vector<int> &k_indices,
                                              std::vector<float> &k_sqr_dists, unsigned int max_nn) const
{
  assert (point_representation_->isValid (point) && "Invalid (NaN, Inf) point coordinates given to radiusSearch!");

  std::vector<float> query (dim_);
  point_representation_->vectorize (static_cast<PointT> (point), query);

  // Has max_nn been set properly?
  if (max_nn == 0 || max_nn > static_cast<unsigned int> (total_nr_points_))
    max_nn = total_nr_points_;

  cache_radius_search_results.clear();

  int neighbors_in_radius = flann_index_->radiusSearch (&query[0], radius, cache_radius_search_results,  param_radius_);
  neighbors_in_radius = std::min<unsigned int>(neighbors_in_radius, max_nn);

  k_indices.resize(neighbors_in_radius);
  k_sqr_dists.resize(neighbors_in_radius);

  if (indices_)
  {
	  for (int i = 0; i < neighbors_in_radius; ++i)
	  {
		  k_sqr_dists[i] = cache_radius_search_results[i].second;
		  k_indices[i] =	indices_->at(cache_radius_search_results[i].first);
	  }
  }
  else
  {
	  for (int i = 0; i < neighbors_in_radius; ++i)
	  {
		  k_sqr_dists[i] = cache_radius_search_results[i].second;
		  k_indices[i] = cache_radius_search_results[i].first;
	  }
  }

  return (neighbors_in_radius);
}

///////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT, typename Dist> void 
pcl::KdTreeFLANN<PointT, Dist>::cleanup ()
{
  // Data array cleanup
  if (indices_)
    indices_.reset ();
}

#if 0
///////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT, typename Dist> void 
pcl::KdTreeFLANN<PointT, Dist>::convertCloudToArray (const PointCloud &cloud)
{
  // No point in doing anything if the array is empty
  if (cloud.points.empty ())
  {
    cloud_.reset ();
    return;
  }

  int original_no_of_points = static_cast<int> (cloud.points.size ());

  cloud_.reset (new float[original_no_of_points * dim_]);
  float* cloud_ptr = cloud_.get ();
  index_mapping_.reserve (original_no_of_points);
  identity_mapping_ = true;

  for (int cloud_index = 0; cloud_index < original_no_of_points; ++cloud_index)
  {
    // Check if the point is invalid
    if (!point_representation_->isValid (cloud.points[cloud_index]))
    {
      identity_mapping_ = false;
      continue;
    }

    index_mapping_.push_back (cloud_index);

    point_representation_->vectorize (cloud.points[cloud_index], cloud_ptr);
    cloud_ptr += dim_;
  }
}

///////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT, typename Dist> void 
pcl::KdTreeFLANN<PointT, Dist>::convertCloudToArray (const PointCloud &cloud, const std::vector<int> &indices)
{
  // No point in doing anything if the array is empty
  if (cloud.points.empty ())
  {
    cloud_.reset ();
    return;
  }

  int original_no_of_points = static_cast<int> (indices.size ());

  cloud_.reset (new float[original_no_of_points * dim_]);
  float* cloud_ptr = cloud_.get ();
  index_mapping_.reserve (original_no_of_points);
  // its a subcloud -> false
  // true only identity: 
  //     - indices size equals cloud size
  //     - indices only contain values between 0 and cloud.size - 1
  //     - no index is multiple times in the list
  //     => index is complete
  // But we can not guarantee that => identity_mapping_ = false
  identity_mapping_ = false;
  
  for (std::vector<int>::const_iterator iIt = indices.begin (); iIt != indices.end (); ++iIt)
  {
    // Check if the point is invalid
    if (!point_representation_->isValid (cloud.points[*iIt]))
      continue;

    // map from 0 - N -> indices [0] - indices [N]
    index_mapping_.push_back (*iIt);  // If the returned index should be for the indices vector
    
    point_representation_->vectorize (cloud.points[*iIt], cloud_ptr);
    cloud_ptr += dim_;
  }
}
#endif

#define PCL_INSTANTIATE_KdTreeFLANN(T) template class PCL_EXPORTS pcl::KdTreeFLANN<T>;

#endif  //#ifndef _PCL_KDTREE_KDTREE_IMPL_FLANN_H_

