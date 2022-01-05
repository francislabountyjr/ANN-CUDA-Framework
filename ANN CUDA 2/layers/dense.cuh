#pragma once

#include "layer.cuh"

namespace cudl
{
	class Dense : public Layer
	{
	public:
		Dense(std::string name, int output_size);
		virtual ~Dense();

		virtual Blob<float>* forward(Blob<float>* input);
		virtual Blob<float>* backward(Blob<float>* grad_output);

	private:
		void fwd_initialize(Blob<float>* input);
		void bwd_initialize(Blob<float>* grad_output);

		int input_size_ = 0;
		int output_size_ = 0;

		float* d_one_vec = nullptr;
	};
}