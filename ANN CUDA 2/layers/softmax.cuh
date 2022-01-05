#pragma once

#include <assert.h>

#include "layer.cuh"

namespace cudl
{
	class Softmax : public Layer
	{
	public:
		Softmax(std::string name);
		virtual ~Softmax();

		virtual Blob<float>* forward(Blob<float>* input);
		virtual Blob<float>* backward(Blob<float>* target);

		float get_loss(Blob<float>* target);
		int get_accuracy(Blob<float>* target);

	private:
		void fwd_initialize(Blob<float>* input);
		void bwd_initialize(Blob<float>* target);

		CrossEntropyLoss loss_;
	};
}