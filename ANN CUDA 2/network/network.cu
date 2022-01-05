#include "network.cuh"

using namespace cudl;

Network::Network()
{
	// do nothing
}

Network::~Network()
{
	for (auto layer : layers_)
	{
		delete layer;
	}

	if (cuda_ != nullptr)
	{
		delete cuda_;
	}
}

void Network::add_layer(Layer* layer)
{
	layers_.push_back(layer);

	// tag layer to stop gradient if it is the first layer
	if (layers_.size() == 1)
	{
		layers_.at(0)->set_gradient_stop();
	}
}

Blob<float>* Network::forward(Blob<float>* input)
{
	output_ = input;

	nvtxRangePushA("Forward");
	for (auto layer : layers_)
	{
#if (DEBUG_FORWARD)
		std::cout << "[Forward][" << std::setw(7) << layer->get_name() << "]\t(" << output_->n() << ", " << output_->c() << ", " << output_->h() << ", " << output_->w() << ")\t";
#endif // DEBUG_FORWARD

		layer->fwd_initialize(output_);
		output_ = layer->forward(output_);

#if (DEBUG_FORWARD)
		std::cout << "--> (" << output_->n() << ", " << output_->c() << ", " << output_->h() << ", " << output_->w() << ")\n";
		checkCudaErrors(cudaDeviceSynchronize());

#if (DEBUG_FORWARD > 1)
		output_->print("output", true);

		if (phase_ == inference)
		{
			getchar();
		}
#endif
#endif // DEBUG_FORWARD
	}
	nvtxRangePop();

	return output_;
}

void Network::backward(Blob<float>* target)
{
	Blob<float>* gradient = target;

	if (phase_ == inference)
	{
		return;
	}

	// back propagation (update weights internally)
	nvtxRangePushA("Backward");
	for (auto layer = layers_.rbegin(); layer != layers_.rend(); layer++)
	{
		// getting back propagation status with gradient size
#if (DEBUG_BACKWARD)
		std::cout << "[Backward][" << std::setw(7) << (*layer)->get_name() << "]\t(" << gradient->n() << ", " << gradient->c() << ", " << gradient->h() << ", " << gradient->w() << ")\t";
#endif // DEBUG_BACKWARD

		(*layer)->bwd_initialize(gradient);
		gradient = (*layer)->backward(gradient);

#if (DEBUG_BACKWARD)
		std::cout << "--> (" << gradient->n() << ", " << gradient->c() << ", " << gradient->h() << ", " << gradient->w() << ")\n";
		checkCudaErrors(cudaDeviceSynchronize());

#if (DEBUG_BACKWARD > 1)
		gradient->print((*layer)->get_name() + "::dx", true);
		getchar();
#endif
#endif // DEBUG_BACKWARD
	}
	nvtxRangePop();
}

void Network::update(float learning_rate)
{
	if (phase_ == inference)
	{
		return;
	}

#if (DEBUG_UPDATE)
	std::cout << "Start update...lr = " << learning_rate << '\n';
#endif // DEBUG_UPDATE

	nvtxRangePushA("Update");
	for (auto layer : layers_)
	{
		// pass if no parameters
		if (layer->weights_ == nullptr || layer->grad_weights_ == nullptr || layer->biases_ == nullptr || layer->grad_biases_ == nullptr)
		{
			continue;
		}

		layer->update_weights_biases(learning_rate);
	}
	nvtxRangePop();
}

int Network::load_pretrain(std::string& parameter_location)
{
	for (auto layer : layers_)
	{
		layer->set_parameter_directory(parameter_location);
		layer->set_load_pretrain();
	}

	return 0;
}

int Network::write_file(std::string& parameter_location)
{
	std::cout << "...Storing Weights...\n";

	for (auto layer : layers_)
	{
		layer->set_parameter_directory(parameter_location);
		int err = layer->save_parameter();

		if (err != 0)
		{
			std::cout << "-> error code: " << err << '\n';
			exit(err);
		}
	}

	return 0;
}

float Network::loss(Blob<float>* target)
{
	Layer* layer = layers_.back();
	return layer->get_loss(target);
}

int Network::get_accuracy(Blob<float>* target)
{
	Layer* layer = layers_.back();
	return layer->get_accuracy(target);
}

// 1. initialize cuda resource container
// 2. register the resource container to all layers
void Network::cuda()
{
	cuda_ = new CudaContext();

	std::cout << "...Model Configuration...\n";

	for (auto layer : layers_)
	{
		std::cout << "CUDA: " << layer->get_name() << '\n';
		layer->set_cuda_context(cuda_);
	}
}

void Network::train()
{
	phase_ = training;

	// unfreeze all layers
	for (auto layer : layers_)
	{
		layer->unfreeze();
	}
}

void Network::test()
{
	phase_ = inference;

	// freeze all layers
	for (auto layer : layers_)
	{
		layer->freeze();
	}
}

std::vector<Layer*> Network::layers()
{
	return std::vector<Layer*>();
}