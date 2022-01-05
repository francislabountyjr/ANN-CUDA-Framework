#pragma once

#include <vector>

#include "layer.cuh"

namespace cudl
{
	class Conv2D : public Layer
	{
	public:
		Conv2D(std::string name, int out_channels, int kernel_size, int stride = 1, int padding = 0, int dilation = 1);
		virtual ~Conv2D();

		virtual Blob<float>* forward(Blob<float>* input);
		virtual Blob<float>* backward(Blob<float>* grad_output);

	private:
		void fwd_initialize(Blob<float>* input);
		void bwd_initialize(Blob<float>* grad_output);

		int out_channels_;
		int kernel_size_;
		int stride_;
		int padding_;
		int dilation_;

		std::array<int, 4> output_size_;

		// convolution
		cudnnConvolutionDescriptor_t conv_desc_;

		cudnnConvolutionFwdAlgo_t conv_fwd_algo_;
		cudnnConvolutionBwdDataAlgo_t conv_bwd_data_algo_;
		cudnnConvolutionBwdFilterAlgo_t conv_bwd_filter_algo_;

		size_t workspace_size_ = 0;
		float* d_workspace_ = nullptr;
		virtual void set_workspace();
	};
}