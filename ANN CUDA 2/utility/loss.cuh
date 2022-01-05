#pragma once

#include "blob.cuh"
#include "helper.cuh"

namespace cudl
{
	class CrossEntropyLoss
	{
	public:
		CrossEntropyLoss();
		~CrossEntropyLoss();

		float loss(Blob<float>* predict, Blob<float>* target);
		float accuracy(Blob<float>* predict, Blob<float>* target);

	private:
		// reduced loss
		float h_loss_ = 0.f;
		float* d_loss_ = nullptr;

		float* d_workspace_ = nullptr;
		void init_workspace(int batch_size);
	};
}