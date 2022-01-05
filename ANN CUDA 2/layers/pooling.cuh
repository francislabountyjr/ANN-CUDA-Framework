#pragma once

#include "layer.cuh"

namespace cudl
{
	class Pooling : public Layer
	{
	public:
		Pooling(std::string name, int kernel_size, int padding, int stride, cudnnPoolingMode_t mode);
		virtual ~Pooling();

		virtual Blob<float>* forward(Blob<float>* input);
		virtual Blob<float>* backward(Blob<float>* grad_output);

	private:
		void fwd_initialize(Blob<float>* input);
		void bwd_initialize(Blob<float>* grad_output);

		int kernel_size_;
		int padding_;
		int stride_;
		cudnnPoolingMode_t mode_;

		std::array<int, 4> output_size_;

		cudnnPoolingDescriptor_t pool_desc_;
	};
}