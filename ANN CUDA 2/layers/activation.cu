#include "activation.cuh"

using namespace cudl;

Activation::Activation(std::string name, cudnnActivationMode_t mode, float coef)
{
	name_ = name;
	act_mode_ = mode;
	act_coef_ = coef;

	cudnnCreateActivationDescriptor(&act_desc_);
	cudnnSetActivationDescriptor(act_desc_, act_mode_, CUDNN_PROPAGATE_NAN, act_coef_);
}

Activation::~Activation()
{
	cudnnDestroyActivationDescriptor(act_desc_);
}

Blob<float>* Activation::forward(Blob<float>* input)
{
	cudnnActivationForward(
		cuda_->cudnn(),
		act_desc_,
		&cuda_->one,
		input_desc_, input->cuda(),
		&cuda_->zero,
		output_desc_, output_->cuda()
		);

	return output_;
}

Blob<float>* Activation::backward(Blob<float>* grad_output)
{
	cudnnActivationBackward(
		cuda_->cudnn(),
		act_desc_,
		&cuda_->one,
		output_desc_, output_->cuda(),
		output_desc_, grad_output->cuda(),
		input_desc_, input_->cuda(),
		&cuda_->zero,
		input_desc_, grad_input_->cuda()
	);

	return grad_input_;
}

void Activation::fwd_initialize(Blob<float>* input)
{
	if (input_ == nullptr || batch_size_ != input->n())
	{
		input_ = input;
		input_desc_ = input->tensor();
		batch_size_ = input->n();

		if (output_ == nullptr)
		{
			output_ = new Blob<float>(input->shape());
		}
		else
		{
			output_->reset(input->shape());
		}

		output_desc_ = output_->tensor();
	}
}

void Activation::bwd_initialize(Blob<float>* grad_output)
{
	if (grad_input_ == nullptr || batch_size_ != grad_output->n())
	{
		grad_output_ = grad_output;

		if (grad_input_ == nullptr)
		{
			grad_input_ = new Blob<float>(input_->shape());
		}
		else
		{
			grad_input_->reset(input_->shape());
		}
	}
}
