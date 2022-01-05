#pragma once

#include <vector>
#include <iomanip>
#include <nvtx3/nvToolsExt.h>

#include "helper.cuh"
#include "loss.cuh"
#include "layer.cuh"

namespace cudl
{
	typedef enum
	{
		training,
		inference
	} WorkloadType;

	class Network
	{
	public:
		Network();
		~Network();

		void add_layer(Layer* layer);

		Blob<float>* forward(Blob<float>* input);
		void backward(Blob<float>* target = nullptr);
		void update(float learning_rate = 0.02f);

		int load_pretrain(std::string& parameter_location);
		int write_file(std::string& parameter_location);

		float loss(Blob<float>* target);
		int get_accuracy(Blob<float>* target);

		void cuda();
		void train();
		void test();

		std::vector<Layer*> layers();

		Blob<float>* output_;

	private:
		std::vector<Layer*> layers_;

		CudaContext* cuda_ = nullptr;

		WorkloadType phase_ = inference;
	};
}