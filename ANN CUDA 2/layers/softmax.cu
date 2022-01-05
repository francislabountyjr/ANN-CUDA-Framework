#include "softmax.cuh"

using namespace cudl;

Softmax::Softmax(std::string name)
{
	name_ = name;
}

Softmax::~Softmax()
{
	// do nothing
}

Blob<float>* Softmax::forward(Blob<float>* input)
{
#if (DEBUG_SOFTMAX & 0x01)
	std::cout << name_ << "[FORWARD]\n";
	input_->print(name_ + "::input", true, input->n());
#endif // DEBUG_SOFTMAX

	checkCudnnErrors(cudnnSoftmaxForward(cuda_->cudnn(), CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
		&cuda_->one, input_desc_, input->cuda(),
		&cuda_->zero, output_desc_, output_->cuda()));

#if (DEBUG_SOFTMAX & 0x01)
	output_->print(name_ + "::output", true, input->n());
#endif // DEBUG_SOFTMAX

	return output_;
}

Blob<float>* Softmax::backward(Blob<float>* target)
{
	// set grad_input_ as predict
	checkCudaErrors(cudaMemcpyAsync(grad_input_->cuda(), output_->cuda(), output_->buf_size(), cudaMemcpyDeviceToDevice));

	// set grad_input_ = predict - target
	checkCublasErrors(cublasSaxpy(cuda_->cublas(), target->len(),
		&cuda_->minus_one, target->cuda(), 1,
		grad_input_->cuda(), 1));

	// normalize the grad_output by the batch size
	int grad_output_size = target->n() * target->c() * target->h() * target->w();
	float scale = 1.f / static_cast<float>(target->n());

	checkCublasErrors(cublasSscal(cuda_->cublas(), grad_output_size, &scale, grad_input_->cuda(), 1));

#if (DEBUG_SOFTMAX & 0x02)
	std::cout << name_ << "[BACKWARD]\n";
	input_->print(name_ + "::input", true);
	output_->print(name_ + "::predict", true);
	target->print(name_ + "::y", true, target->n());
	grad_input_->print(name_ + "::dx", true, target->n());
#endif // DEBUG_SOFTMAX

	return grad_input_;
}

float Softmax::get_loss(Blob<float>* target)
{
	return loss_.loss(output_, target);
}

int Softmax::get_accuracy(Blob<float>* target)
{
	int batch_size = output_->n();
	int output_size = output_->size();

	assert(batch_size == target->n());
	assert(output_size == target->size());

	float* h_output, * h_target;
	int idx_output, idx_target;
	int hit_count = 0;

	// get predictions and targets
	h_output = output_->to(host);
	h_target = target->to(host);

	for (int b = 0; b < batch_size; b++)
	{
		idx_output = 0;
		idx_target = 0;

		for (int i = 1; i < 10; i++)
		{
			if (h_output[b * output_size + i] > h_output[b * output_size + idx_output])
			{
				idx_output = i;
			}
			
			if (h_target[b * output_size + i] > h_target[b * output_size + idx_target])
			{
				idx_target = i;
			}
		}

		if (idx_output == idx_target)
		{
			hit_count++;
		}
	}

	return hit_count;
}

void Softmax::fwd_initialize(Blob<float>* input)
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

void Softmax::bwd_initialize(Blob<float>* target)
{
	if (grad_input_ == nullptr || batch_size_ != target->n())
	{
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