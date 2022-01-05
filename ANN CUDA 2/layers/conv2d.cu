#include "conv2d.cuh"

using namespace cudl;

Conv2D::Conv2D(std::string name, int out_channels, int kernel_size, int stride, int padding, int dilation)
	:out_channels_(out_channels),
	kernel_size_(kernel_size),
	stride_(stride),
	padding_(padding),
	dilation_(dilation)
{
	name_ = name;

	// create cudnn container handles
	cudnnCreateFilterDescriptor(&filter_desc_);

	cudnnCreateConvolutionDescriptor(&conv_desc_);
	checkCudnnErrors(cudnnSetConvolution2dDescriptor(conv_desc_, padding_, padding_,
		stride_, stride_, dilation_, dilation_, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

	// set cudnn convolution math type
	checkCudnnErrors(cudnnSetConvolutionMathType(conv_desc_, CUDNN_DEFAULT_MATH));

	d_workspace_ = nullptr;
}

Conv2D::~Conv2D()
{
	// destroy cudnn container resources
	cudnnDestroyFilterDescriptor(filter_desc_);
	cudnnDestroyConvolutionDescriptor(conv_desc_);

	// terminate internal blobs
	if (d_workspace_ != nullptr)
	{
		cudaFree(d_workspace_);
		d_workspace_ = nullptr;
	}
}

Blob<float>* Conv2D::forward(Blob<float>* input)
{
	checkCudnnErrors(cudnnConvolutionForward(cuda_->cudnn(),
		&cuda_->one, input_desc_, input_->cuda(),
		filter_desc_, weights_->cuda(), conv_desc_, conv_fwd_algo_, d_workspace_, workspace_size_,
		&cuda_->zero, output_desc_, output_->cuda()));

	checkCudnnErrors(cudnnAddTensor(cuda_->cudnn(),
		&cuda_->one, bias_desc_, biases_->cuda(),
		&cuda_->one, output_desc_, output_->cuda()));

#if (DEBUG_CONV & 0x01)
	input_->print(name_ + "::input", true, input_->n(), 28);
	weights_->print(name_ + "::weight", true);
	biases_->print(name_ + "::bias", true);
	output_->print(name_ + "::output", true);
#endif // DEBUG_CONV

	return output_;
}

Blob<float>* Conv2D::backward(Blob<float>* grad_output)
{
	// gradients of biases
	checkCudnnErrors(cudnnConvolutionBackwardBias(cuda_->cudnn(),
		&cuda_->one,
		output_desc_, grad_output->cuda(),
		&cuda_->zero,
		bias_desc_, grad_biases_->cuda()));

	// gradients of weights
	checkCudnnErrors(cudnnConvolutionBackwardFilter(cuda_->cudnn(),
		&cuda_->one,
		input_desc_, input_->cuda(),
		output_desc_, grad_output_->cuda(),
		conv_desc_, conv_bwd_filter_algo_, d_workspace_, workspace_size_,
		&cuda_->zero,
		filter_desc_, grad_weights_->cuda()));

	// gradients of input data
	if (!gradient_stop_)
	{
		checkCudnnErrors(cudnnConvolutionBackwardData(cuda_->cudnn(),
			&cuda_->one,
			filter_desc_, weights_->cuda(),
			output_desc_, grad_output->cuda(),
			conv_desc_, conv_bwd_data_algo_, d_workspace_, workspace_size_,
			&cuda_->zero,
			input_desc_, grad_input_->cuda()));
	}

#if (DEBUG_CONV & 0x02)
		std::cout << name_ << "[BACKWARD]\n";
		grad_output->print(name_ + "::gradients", true;
		grad_weights_->print(name_ + "::gfilter", true);
		grad_biases_->print(name_ + "::gbias", true);
		if (!gradient_stop_)
		{
			grad_input_->print(name_ + "::gdata", true);
		}
#endif // DEBUG_CONV

#if (DEBUG_CONV & 0x04)
		std::cout << name_ << "[BACKWARD]\n";
		grad_output->print(name_ + "::gradients", true;
		grad_weights_->print(name_ + "::gfilter", true);
		grad_biases_->print(name_ + "::gbias", true);
		if (!gradient_stop_)
		{
			grad_input_->print(name_ + "::gdata", true);
		}
#endif // DEBUG_CONV

		return grad_input_;
}

void Conv2D::fwd_initialize(Blob<float>* input)
{
	// initialize weights and bias
	if (weights_ == nullptr)
	{
		// initialize container handles
	
		checkCudnnErrors(cudnnSetFilter4dDescriptor(filter_desc_, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
			out_channels_, input->c(), kernel_size_, kernel_size_));

		weights_ = new Blob<float>(out_channels_, input->c(), kernel_size_, kernel_size_);
		biases_ = new Blob<float>(1, out_channels_);
		bias_desc_ = biases_->tensor();
	}

	// initialize input and output
	if (input_ == nullptr || batch_size_ != input->n())
	{
		// initialize input
		input_ = input;
		input_desc_ = input->tensor();
		batch_size_ = input->n();

		// initialize output
		checkCudnnErrors(cudnnGetConvolution2dForwardOutputDim(conv_desc_, input_desc_, filter_desc_,
			&output_size_[0], &output_size_[1], &output_size_[2], &output_size_[3]));

		if (output_ == nullptr)
		{
			output_ = new Blob<float>(output_size_);
		}
		else
		{
			output_->reset(output_size_);
		}

		output_desc_ = output_->tensor();

		// initialize workspace for cudnn
		set_workspace();

		// initialize weights
		if (load_pretrain_ && !freeze_)
		{
			if (load_parameter())
			{
				std::cout << "error occured loading weights for " << name_ << '\n';
				exit(-1);
			}
		}
		else if (!freeze_)
		{
			init_weight_bias();
		}
		else
		{
			// do nothing
		}
	}
}

void Conv2D::bwd_initialize(Blob<float>* grad_output)
{
	if (grad_weights_ == nullptr)
	{
		grad_weights_ = new Blob<float>(weights_->shape());
		grad_biases_ = new Blob<float>(1, biases_->c());
	}

	// initialize grad_output back-propagation space
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

void Conv2D::set_workspace()
{
	size_t temp_size = 0;

	// forward
#if CUDNN_MAJOR >= 7
	std::vector<cudnnConvolutionFwdAlgoPerf_t> fwd_algo_perf_results(CUDNN_CONVOLUTION_FWD_ALGO_COUNT);
	std::vector<cudnnConvolutionBwdFilterAlgoPerf_t> bwd_filter_algo_perf_results(CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT);
	std::vector<cudnnConvolutionBwdDataAlgoPerf_t> bwd_data_algo_perf_results(CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT);

	int algo_max_count;
	int returnedAlgoCount = 0;
	checkCudnnErrors(cudnnGetConvolutionForwardAlgorithmMaxCount(cuda_->cudnn(), &algo_max_count));
#if (DEBUG_FIND_ALGO & 1)
	std::cout << this->name_ << ": Available Algorithm Count [FWD]: " << algo_max_count << '\n';

	checkCudnnErrors(cudnnFindConvolutionForwardAlgorithm(cuda_->cudnn(),
		input_desc_, filter_desc_, conv_desc_, output_desc_,
		algo_max_count, &returnedAlgoCount, &fwd_algo_perf_results[0]));

	std::cout << "returned algo_count: " << returnedAlgoCount << '\n';

	for (int i = 0; i < returnedAlgoCount; i++)
	{
		std::cout << "fwd algo[" << i << "] time: " << fwd_algo_perf_results[i].time << ", memory: " << fwd_algo_perf_results[i].memory << '\n';
#else
	checkCudnnErrors(cudnnGetConvolutionForwardAlgorithm_v7(cuda_->cudnn(),
		input_desc_, filter_desc_, conv_desc_, output_desc_,
		algo_max_count, &returnedAlgoCount, &fwd_algo_perf_results[0]));
#endif
	// choose the fastest algorithm
	conv_fwd_algo_ = fwd_algo_perf_results[0].algo;
#else
	checkCudnnErrors(cudnnGetConvolutionForwardAlgorithm(cuda_->cudnn(),
		input_desc_, filter_desc_, conv_desc_, output_desc_,
		CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &conv_fwd_algo_));
#endif
	checkCudnnErrors(cudnnGetConvolutionForwardWorkspaceSize(cuda_->cudnn(),
		input_desc_, filter_desc_, conv_desc_, output_desc_,
		conv_fwd_algo_, &temp_size));

	workspace_size_ = std::max(workspace_size_, temp_size);

	// bwd - filter
#if CUDNN_MAJOR >= 7
	checkCudnnErrors(cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(cuda_->cudnn(), &algo_max_count));
#if (DEBUG_FIND_ALGO & 1)
	std::cout << this->name_ << ": Available Algorithm Count [BWD-filter]: " << algo_max_count << '\n';

	checkCudnnErrors(cudnnFindConvolutionBackwardFilterAlgorithm(cuda_->cudnn(),
		input_desc_, output_desc_, conv_desc_, filter_desc_,
		algo_max_count, &returnedAlgoCount, &bwd_filter_algo_perf_results[0]));

	std::cout << "returned algo_count: " << returnedAlgoCount << '\n';

	for (int i = 0; i < returnedAlgoCount; i++)
	{
		std::cout << "bwd filter algo[" << i << "] time: " << bwd_filter_algo_perf_results[i].time << ", memory: " << bwd_filter_algo_perf_results[i].memory << '\n';
#else
	checkCudnnErrors(cudnnGetConvolutionBackwardFilterAlgorithm_v7(cuda_->cudnn(),
		input_desc_, output_desc_, conv_desc_, filter_desc_,
		algo_max_count, &returnedAlgoCount, &bwd_filter_algo_perf_results[0]));
#endif
	// choose the fastest algorithm
	conv_bwd_filter_algo_ = bwd_filter_algo_perf_results[0].algo;
#else
	checkCudnnErrors(cudnnGetConvolutionBackwardFilterAlgorithm(cuda_->cudnn(),
		input_desc_, output_desc_, conv_desc_, filter_desc_,
		CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, &conv_bwd_filter_algo_));
#endif
	checkCudnnErrors(cudnnGetConvolutionBackwardFilterWorkspaceSize(cuda_->cudnn(),
		input_desc_, output_desc_, conv_desc_, filter_desc_,
		conv_bwd_filter_algo_, &temp_size));

	workspace_size_ = std::max(workspace_size_, temp_size);

	// bwd - data
#if CUDNN_MAJOR >= 7
	checkCudnnErrors(cudnnGetConvolutionBackwardDataAlgorithmMaxCount(cuda_->cudnn(), &algo_max_count));
#if (DEBUG_FIND_ALGO & 1)
	std::cout << this->name_ << ": Available Algorithm Count [BWD-data]: " << algo_max_count << '\n';

	checkCudnnErrors(cudnnFindConvolutionBackwardDataAlgorithm(cuda_->cudnn(),
		filter_desc_, output_desc_, conv_desc_, input_desc_,
		algo_max_count, &returnedAlgoCount, &bwd_data_algo_perf_results[0]));

	std::cout << "returned algo_count: " << returnedAlgoCount << '\n';

	for (int i = 0; i < returnedAlgoCount; i++)
	{
		std::cout << "bwd data algo[" << i << "] time: " << bwd_data_algo_perf_results[i].time << ", memory: " << bwd_data_algo_perf_results[i].memory << '\n';
#else
	checkCudnnErrors(cudnnGetConvolutionBackwardDataAlgorithm_v7(cuda_->cudnn(),
		filter_desc_, output_desc_, conv_desc_, input_desc_,
		algo_max_count, &returnedAlgoCount, &bwd_data_algo_perf_results[0]));
#endif
	// choose the fastest algorithm
	conv_bwd_data_algo_ = bwd_data_algo_perf_results[0].algo;
#else
	checkCudnnErrors(cudnnGetConvolutionBackwardDataAlgorithm(cuda_->cudnn(),
		filter_desc_, output_desc_, conv_desc_, input_desc_,
		CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, &conv_bwd_data_algo_));
#endif
	checkCudnnErrors(cudnnGetConvolutionBackwardDataWorkspaceSize(cuda_->cudnn(),
		filter_desc_, output_desc_, conv_desc_, input_desc_,
		conv_bwd_data_algo_, &temp_size));

	workspace_size_ = std::max(workspace_size_, temp_size);

	if (workspace_size_ > 0)
	{
		if (d_workspace_ != nullptr)
		{
			checkCudaErrors(cudaFree(d_workspace_));
		}

		checkCudaErrors(cudaMalloc((void**)&d_workspace_, workspace_size_));
	}
}