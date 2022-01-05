#pragma once

#include <string>

#include <cublas_v2.h>
#include <cudnn.h>

#include "blob.cuh"
#include "loss.cuh"
#include "helper.cuh"

namespace cudl
{
	class Layer
	{
	public:
		Layer();
		virtual ~Layer();

		virtual Blob<float>* forward(Blob<float>* input) = 0;
		virtual Blob<float>* backward(Blob<float>* grad_input) = 0;

		std::string get_name() { return name_; }

		void set_parameter_directory(std::string& parameter_location) { parameter_location_ = parameter_location; }

		virtual float get_loss(Blob<float>* target);
		virtual int get_accuracy(Blob<float>* target);

		void set_cuda_context(CudaContext* context) { cuda_ = context; }

		void set_load_pretrain() { load_pretrain_ = true; }
		void set_gradient_stop() { gradient_stop_ = true; }

		// weight freeze or unfreeze
		void freeze() { freeze_ = true; }
		void unfreeze() { freeze_ = false; }
		
	protected:
		virtual void fwd_initialize(Blob<float>* input) = 0;
		virtual void bwd_initialize(Blob<float>* grad_output) = 0;

		// layer name
		std::string name_;

		// tensor descriptor for input and output tensors
		cudnnTensorDescriptor_t input_desc_;
		cudnnTensorDescriptor_t output_desc_;

		// weight and bias descriptor
		cudnnFilterDescriptor_t filter_desc_;
		cudnnTensorDescriptor_t bias_desc_;

		// output memory
		Blob<float>* input_ = nullptr; // x
		Blob<float>* output_ = nullptr; // y
		Blob<float>* grad_input_ = nullptr; // dx
		Blob<float>* grad_output_ = nullptr; //dy

		// master weights and bias
		bool freeze_ = false; // control parameter updates
		Blob<float>* weights_ = nullptr; // w
		Blob<float>* biases_ = nullptr; // b
		Blob<float>* grad_weights_ = nullptr; // dw
		Blob<float>* grad_biases_ = nullptr; //db

		int batch_size_ = 0; // mini-batch size

		// initialize weights along with the input size
		void init_weight_bias(unsigned int seed = 0);
		void update_weights_biases(float learning_rate);

		// cuda handle container
		CudaContext* cuda_ = nullptr;

		// folder to save parameters in
		std::string parameter_location_;

		// pretrain parameters
		bool load_pretrain_ = false;
		int load_parameter();
		int save_parameter();

		// gradient stop tagging
		bool gradient_stop_ = false;

		friend class Network;
	};
}