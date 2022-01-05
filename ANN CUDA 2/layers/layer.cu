#include "layer.cuh"

#include <random>
#include <cassert>
#include <math.h>
#include <algorithm>
#include <sstream>
#include <fstream>
#include <iostream>

#include <curand.h>

using namespace cudl;

// Layer Definition
Layer::Layer()
{
	// do nothing
}

Layer::~Layer()
{
#if (DEBUG_FORWARD > 0 || DEBUG_BACKWARD > 0)
	std::cout << "Destroy Layer: " << name_ << '\n';
#endif
	if (output_ != nullptr) { delete output_; output_ = nullptr; }
	if (grad_input_ != nullptr) { delete grad_input_; grad_input_ = nullptr; }

	if (weights_ != nullptr) { delete weights_; weights_ = nullptr; }
	if (biases_ != nullptr) { delete biases_; biases_ = nullptr; }

	if (grad_weights_ != nullptr) { delete grad_weights_; grad_weights_ = nullptr; }
	if (grad_biases_ != nullptr) { delete grad_biases_; grad_biases_ = nullptr; }
}

float Layer::get_loss(Blob<float>* target)
{
	assert("No Loss layer - no loss" && false);
	return EXIT_FAILURE;
}

int Layer::get_accuracy(Blob<float>* target)
{
	assert("No Loss layer - cannot estimate accuracy" && false);
	return EXIT_FAILURE;
}

void Layer::init_weight_bias(unsigned int seed)
{
	checkCudaErrors(cudaDeviceSynchronize());

	if (weights_ == nullptr || biases_ == nullptr)
	{
		return;
	}

	// create random network
	std::random_device rd;
	std::mt19937 gen(seed == 0 ? rd() : static_cast<unsigned int>(seed));

	// he uniform distribution
	float range = sqrt(6.f / input_->size()); // he initializaiton
	std::uniform_real_distribution<> dis(-range, range);

	for (int i = 0; i < weights_->len(); i++)
	{
		weights_->ptr()[i] = static_cast<float>(dis(gen));
	}

	for (int i = 0; i < biases_->len(); i++)
	{
		biases_->ptr()[i] = 0.f;
	}

	// copy initialized values to the device
	weights_->to(DeviceType::cuda);
	biases_->to(DeviceType::cuda);

	std::cout << "..initialized " << name_ << " layer..\n";
}

void Layer::update_weights_biases(float learning_rate)
{
	float eps = -1.f * learning_rate;

	if (weights_ != nullptr && grad_weights_ != nullptr)
	{
#if (DEBUG_UPDATE)
		weights_->print(name_ + "::weights (before update)", true);
		grad_weights_->print(name_ + "::gweights", true);
#endif // DEBUG_UPDATE

		// w = w + eps * dw
		checkCublasErrors(cublasSaxpy(cuda_->cublas(),
									  weights_->len(),
									  &eps,
									  grad_weights_->cuda(), 1,
									  weights_->cuda(), 1));

#if (DEBUG_UPDATE)
		weights_->print(name_ + "::weights (after update)", true);
#endif  // DEBUG_UPDATE
	}

	if (biases_ != nullptr && grad_biases_ != nullptr)
	{
#if (DEBUG_UPDATE)
		biases_->print(name_ + "::biases (before update)", true);
		grad_biases_->print(name_ + "::gbiases", true);
#endif  // DEBUG_UPDATE

		// b = b + eps * db
		checkCublasErrors(cublasSaxpy(cuda_->cublas(),
									  biases_->len(),
									  &eps,
									  grad_biases_->cuda(), 1,
									  biases_->cuda(), 1));

#if (DEBUG_UPDATE)
		biases_->print(name_ + "::biases (after update)", true);
#endif  // DEBUG_UPDATE
	}
}

int Layer::load_parameter()
{
	std::stringstream filename_weights, filename_biases;

	// load pretrained weights and biases
	filename_weights << parameter_location_ << '/' << name_ << ".bin";
	if (weights_->file_read(filename_weights.str()))
	{
		return -1;
	}

	filename_biases << parameter_location_ << '/' << name_ << ".bias.bin";
	if (biases_->file_read(filename_biases.str()))
	{
		return -2;
	}

	std::cout << "..loaded " << name_ << " pretrained weights and biases..\n";

	return 0;
}

int Layer::save_parameter()
{
	std::stringstream filename_weights, filename_biases;

	std::cout << "..saving " << name_ << " weights and biases..\n";

	// write weights file
	if (weights_)
	{
		filename_weights << parameter_location_ << '/' << name_ << ".bin";
		if (weights_->file_write(filename_weights.str()))
		{
			return -1;
		}
	}

	// write bias file
	if (biases_)
	{
		filename_biases << parameter_location_ << '/' << name_ << ".bias.bin";
		if (biases_->file_write(filename_biases.str()))
		{
			return -2;
		}
	}

	std::cout << "..done..\n";

	return 0;
}
