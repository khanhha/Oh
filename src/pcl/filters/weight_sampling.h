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

#ifndef PCL_FILTERS_WEIGHT_SAMPLING_H_
#define PCL_FILTERS_WEIGHT_SAMPLING_H_

#include <memory>
#include <vector>
#include <Eigen/Dense>
#include <pcl/filters/filter.h>

namespace pcl
{
	template <typename PointT>
	class WeightSampling : public Filter<PointT>
	{
		typedef typename Filter<PointT>::PointCloud PointCloud;

		using Filter<PointT>::filter_name_;
		using Filter<PointT>::input_;
		using Filter<PointT>::indices_;
		using Filter<PointT>::getClassName;

	public:
		typedef std::shared_ptr<WeightSampling<PointT> > Ptr;
		typedef std::shared_ptr<const WeightSampling<PointT> > ConstPtr;

		/** \brief Empty constructor. */
		WeightSampling()
			: resample_percent_(0.5)
			, k_neighbour_search_(10)
			, sigma_(1.5)
		{
			filter_name_ = "WeightSampling";
		}
		/** \brief Destructor. */
		virtual ~WeightSampling()
		{
		}

		void setResamplePercent(float percent = 0.5) { resample_percent_ = percent; }
		void setKNeighbourSearch(int k = 10) { k_neighbour_search_ = k; };
		void setSigma(float sig = 1.5) { sigma_ = sig; };
	public:
		//std::vector<Eigen::Vector3f>   test_sample_points;
		//std::vector<float>			 test_weights;
	protected:
		void
			applyFilter(PointCloud &output);
		float resample_percent_;
		float k_neighbour_search_;
		float sigma_;
	};
}

#ifdef PCL_NO_PRECOMPILE
#include <pcl/filters/impl/weight_sampling.hpp>
#endif

#endif  //#ifndef PCL_FILTERS_UNIFORM_SAMPLING_H_

