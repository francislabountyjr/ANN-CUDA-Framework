#pragma once

#include "layer.cuh"

namespace cudl
{
	class Activation : public Layer
	{
	public:
		Activation(std::string name, cudnnActivationMode_t mode, float coef = 0.f);
		virtual ~Activation();

		virtual Blob<float>* forward(Blob<float>* input);
		virtual Blob<float>* backward(Blob<float>* grad_output);

	private:
		void fwd_initialize(Blob<float>* input);
		void bwd_initialize(Blob<float>* grad_output);

		cudnnActivationDescriptor_t act_desc_;
		cudnnActivationMode_t act_mode_;
		float act_coef_;
	};
}