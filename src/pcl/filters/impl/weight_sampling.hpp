/*
* Software License Agreement (BSD License)
*
*  Copyright (c) 2009, Willow Garage, Inc.
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

#ifndef PCL_FILTERS_WEIGHT_SAMPLING_IMPL_H_
#define PCL_FILTERS_WEIGHT_SAMPLING_IMPL_H_

#include <pcl/common/common.h>
#include <pcl/filters/weight_sampling.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <cstdlib>
#include <random>
#include <fstream>
#include <algorithm>
#include <utility>

template <typename T> std::vector<T> operator+(const std::vector<T>& a, const std::vector<T>& b)
{
	assert(a.size() == b.size());
	std::vector<T> result;
	result.reserve(a.size());
	std::transform(a.begin(), a.end(), b.begin(),
		std::back_inserter(result), std::plus<T>());
	return result;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int computeWeights(pcl::PointCloud<pcl::PointXYZ>::ConstPtr pc, int num_pts, float sigma_sq, int K, std::vector<float>& imp_wt) {

	pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
	kdtree.setInputCloud(pc);
	kdtree.addPointsFromInputCloud();

	float feature_length = 3;
	float normalization_sum = 0.0;

	std::vector<int> pointIdxNKNSearch(K);
	std::vector<float> pointNKNSquaredDistance(K);
	std::vector<float> x_j_feature(feature_length, 0.0);
	std::vector<float> sum_nbr(feature_length, 0.0);

	for (int i = 0; i < num_pts; i++) {

		pcl::PointXYZ searchPoint = pc->points[i];
		float Aij_ew;
		imp_wt[i] = 0.0;
		if (kdtree.nearestKSearch(searchPoint, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
		{
			std::fill(sum_nbr.begin(), sum_nbr.end(), 0.0);
			float total_ew = 0.0;
			for (size_t j = 1; j < pointIdxNKNSearch.size(); ++j)
			{
				int x_j = pointIdxNKNSearch[j];

				Aij_ew = exp(-1.0*(pointNKNSquaredDistance[j] / sigma_sq));
				total_ew += Aij_ew;
				x_j_feature[0] = Aij_ew * pc->points[x_j].x;
				x_j_feature[1] = Aij_ew * pc->points[x_j].y;
				x_j_feature[2] = Aij_ew * pc->points[x_j].z;
				//x_j_feature[3] = Aij_ew * ((float(pc->points[x_j].r)) / 255.0);
				//x_j_feature[4] = Aij_ew * ((float(pc->points[x_j].g)) / 255.0);
				//x_j_feature[5] = Aij_ew * ((float(pc->points[x_j].b)) / 255.0);

				//sum_nbr = operator+(sum_nbr, x_j_feature);
				for (size_t k = 0; k < feature_length; ++k)
					sum_nbr[k] += x_j_feature[k];
			}
				
			//compute norm
			for (size_t k = 0; k < feature_length; ++k)
				sum_nbr[k] /= total_ew;

			float norm_sum = 0.0;
			norm_sum += pow(sum_nbr[0] - searchPoint.x, 2.0);
			norm_sum += pow(sum_nbr[1] - searchPoint.y, 2.0);
			norm_sum += pow(sum_nbr[2] - searchPoint.z, 2.0);
			//norm_sum += pow(sum_nbr[3] - ((float)searchPoint.r) / 255.0, 2.0);
			//norm_sum += pow(sum_nbr[4] - ((float)searchPoint.g) / 255.0, 2.0);
			//norm_sum += pow(sum_nbr[5] - ((float)searchPoint.b) / 255.0, 2.0);
			imp_wt[i] = norm_sum;
		}
		else {
			imp_wt[i] = 0.0;
		}

		normalization_sum += imp_wt[i];
	}

	for (int i = 0; i < num_pts; i++) {
		imp_wt[i] = imp_wt[i] / normalization_sum;
	}

	return(1);
}

std::vector<int> sampleNLargest(std::vector<float> weights, int num_pts, int total_samples) 
{
	total_samples = std::min(total_samples, num_pts);

	std::vector<std::pair<int, float>> weight_idx(weights.size());
	for (size_t i = 0; i < weights.size(); ++i)
		weight_idx[i] = std::make_pair(i, weights[i]);

	std::sort(
		weight_idx.begin(), weight_idx.end(),
		[](const std::pair<int, float> &p1, const std::pair<int, float> &p2) 
		{ 
			return p1.second >= p2.second; 
		});

	std::vector<int> sample_idx(total_samples);
	for (size_t i = 0; i < total_samples; ++i)
		sample_idx[i] = weight_idx[i].first;
	
	return sample_idx;
}

std::vector<int> sampleRandomDistribution(std::vector<float> weights, int num_pts, int total_samples) {

	srand(static_cast <unsigned> (time(0)));
	std::vector<int> samples(total_samples, 0);
	std::vector<float> imp_wt_rs(num_pts, 0.0);

	// calculate rolling sum of weights:
	for (int i = 1; i < num_pts; i++) {
		imp_wt_rs[i] = imp_wt_rs[i - 1] + weights[i - 1];
	}

	for (int i = 0; i < total_samples; i++) {
		float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
		// find bin:
		for (int j = 0; j < num_pts - 1; j++) {
			if (r <= imp_wt_rs[j + 1] && r >= imp_wt_rs[j]) {
				samples[i] = j;
				//printf("r: %f, low: %f, high: %f \n",r,imp_wt_rs[j],imp_wt_rs[j+1]);
				//printf("sampleidx: %d, imp_wt: %0.9f \n ", j, imp_wt[j]);
				break;
			}
		}
		if (r > imp_wt_rs[num_pts - 1]) {
			samples[i] = num_pts - 1;
		}
	}
	return(samples);
}

std::vector<int> sampleNormalDistribution(std::vector<float> weights, int num_pts, int total_samples) {

	srand(static_cast <unsigned> (time(0)));
	std::vector<int> samples(num_pts, 0);
	std::vector<float> imp_wt_rs(num_pts, 0.0);

	std::random_device rd;  //Will be used to obtain a seed for the random number engine
	std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
	std::uniform_real_distribution<> dis(0.0, 1.0);

	std::sort(weights.begin(), weights.end());

	// calculate rolling sum of weights:
	for (int i = 1; i < num_pts; i++) {
		imp_wt_rs[i] = imp_wt_rs[i - 1] + weights[i - 1];
	}

	for (int i = 0; i < total_samples; i++) {
		//float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
		float r = dis(gen);
		// find bin:
		for (int j = 0; j < num_pts - 1; j++) {
			if (r <= imp_wt_rs[j + 1] && r >= imp_wt_rs[j]) {
				samples[i] = j;
				//printf("r: %f, low: %f, high: %f \n",r,imp_wt_rs[j],imp_wt_rs[j+1]);
				//printf("sampleidx: %d, imp_wt: %0.9f \n ", j, imp_wt[j]);
				break;
			}
		}
		if (r > imp_wt_rs[num_pts - 1]) {
			samples[i] = num_pts - 1;
		}
	}
	return(samples);
}


void resamplePC(pcl::PointCloud<pcl::PointXYZ>::ConstPtr pc, pcl::PointCloud<pcl::PointXYZ>& pc_rs, std::vector<int> samples, int total_samples) {
	pc_rs.width = total_samples;
	pc_rs.height = 1;
	pc_rs.points.resize(pc_rs.height * pc_rs.width);

	for (int i = 0; i < total_samples; i++) {
		pc_rs.points[i].x = pc->points[samples[i]].x;
		pc_rs.points[i].y = pc->points[samples[i]].y;
		pc_rs.points[i].z = pc->points[samples[i]].z;
		//pc_rs.points[i].r = pc->points[samples[i]].r;
		//pc_rs.points[i].g = pc->points[samples[i]].g;
		//pc_rs.points[i].b = pc->points[samples[i]].b;
	}
}

template <typename PointT> void
pcl::WeightSampling<PointT>::applyFilter(PointCloud &output)
{
	// Has the input dataset been set already?
	if (!input_)
	{
		PCL_WARN("[pcl::%s::detectKeypoints] No input dataset given!\n", getClassName().c_str());
		output.width = output.height = 0;
		output.points.clear();
		return;
	}

	int num_pts = input_->size();
	// compute importance weights
	std::vector<float> imp_wt(num_pts, 0.0);
	computeWeights(input_, num_pts,  sigma_*sigma_, k_neighbour_search_, imp_wt);
	
	//test_weights = imp_wt;

	int total_samples = resample_percent_ * num_pts;
	std::vector<int> sampleIdx = sampleRandomDistribution(imp_wt, num_pts, total_samples);
	//std::vector<int> sampleIdx = sampleNLargest(imp_wt, num_pts, total_samples);

	resamplePC(input_, output, sampleIdx, total_samples);
}

#define PCL_INSTANTIATE_WeightSampling(T) template class PCL_EXPORTS pcl::WeightSampling<T>;

#endif    // PCL_FILTERS_UNIFORM_SAMPLING_IMPL_H_

