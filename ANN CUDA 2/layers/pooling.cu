#include "pooling.cuh"

using namespace cudl;

Pooling::Pooling(std::string name, int kernel_size, int padding, int stride, cudnnPoolingMode_t mode)
	:kernel_size_(kernel_size),
	padding_(padding),
	stride_(stride),
	mode_(mode)
{
	name_ = name;

	cudnnCreatePoolingDescriptor(&pool_desc_);
	cudnnSetPooling2dDescriptor(pool_desc_, mode_, CUDNN_PROPAGATE_NAN,
		kernel_size_, kernel_size_, padding_, padding_, stride_, stride_);
}

Pooling::~Pooling()
{
	cudnnDestroyPoolingDescriptor(pool_desc_);
}

Blob<float>* Pooling::forward(Blob<float>* input)
{
	cudnnPoolingForward(cuda_->cudnn(), pool_desc_,
		&cuda_->one,
		input_desc_, input_->cuda(),
		&cuda_->zero,
		output_desc_, output_->cuda());

	return output_;
}

Blob<float>* Pooling::backward(Blob<float>* grad_output)
{
	checkCudnnErrors(cudnnPoolingBackward(cuda_->cudnn(), pool_desc_,
		&cuda_->one,
		output_desc_, output_->cuda(),
		output_desc_, grad_output->cuda(),
		input_desc_, input_->cuda(),
		&cuda_->zero,
		input_desc_, grad_input_->cuda()));

	return grad_input_;
}

void Pooling::fwd_initialize(Blob<float>* input)
{
	if (input_ == nullptr || batch_size_ != input->n())
	{
		input_ = input;

		// resource initialization
		input_desc_ = input_->tensor();
		batch_size_ = input->n();

		// setting output
		cudnnGetPooling2dForwardOutputDim(pool_desc_, input_desc_,
			&output_size_[0], &output_size_[1], &output_size_[2], &output_size_[3]);

		if (output_ == nullptr)
		{
			output_ = new Blob<float>(output_size_);
		}
		else
		{
			output_->reset(output_size_);
		}

		output_desc_ = output_->tensor();
	}
}

void Pooling::bwd_initialize(Blob<float>* grad_output)
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