#include <iomanip>
#include <cuda_profiler_api.h>
#include <nvtx3/nvToolsExt.h>

#include "utility/mnist.cuh"
#include "network/network.cuh"
#include "layers/dense.cuh"
#include "layers/activation.cuh"
#include "layers/softmax.cuh"
#include "layers/conv2d.cuh"
#include "layers/pooling.cuh"

using namespace cudl;

int main()
{
	// Network Configuration
	int batch_size_train = 256;
	int num_steps_train = 1600;
	int monitoring_step = 200;

	double learning_rate = 0.02f;
	double lr_decay = 0.00005f;

	bool load_pretrain = false;
	bool file_save = false;
	std::string parameter_location = "";

	int batch_size_test = 10;
	int num_steps_test = 1000;

	// Phase 1: Train
	std::cout << "[TRAIN]\n";

	// Step 1: Load dataset
	MNIST train_data_loader = MNIST("E:/Datasets/mnist");
	train_data_loader.train(batch_size_train, true);

	// Step 2: Model configuration and initialization
	Network model;
	model.add_layer(new Conv2D("conv1", 20, 5));
	model.add_layer(new Activation("relu", CUDNN_ACTIVATION_RELU));
	model.add_layer(new Pooling("pool", 2, 0, 2, CUDNN_POOLING_MAX));
	model.add_layer(new Conv2D("conv1", 50, 5));
	model.add_layer(new Activation("relu", CUDNN_ACTIVATION_RELU));
	model.add_layer(new Pooling("pool", 2, 0, 2, CUDNN_POOLING_MAX));
	model.add_layer(new Dense("dense1", 500));
	model.add_layer(new Activation("relu", CUDNN_ACTIVATION_RELU));
	model.add_layer(new Dense("dense2", 10));
	model.add_layer(new Softmax("softmax"));
	model.cuda();

	if (load_pretrain)
	{
		model.load_pretrain(parameter_location);
	}

	model.train();

	// start Nsight System profile
	cudaProfilerStart();

	// Step 3: Train
	int step = 0;
	Blob<float>* train_data = train_data_loader.get_data();
	Blob<float>* train_target = train_data_loader.get_target();

	train_data_loader.get_batch();

	int tp_count = 0;
	while(step < num_steps_train)
	{
		// nvtx profiling start
		std::string nvtx_message = std::string("step" + std::to_string(step));
		nvtxRangePushA(nvtx_message.c_str());

		// update shared buffer contents
		train_data->to(cuda);
		train_target->to(cuda);

		// forward
		model.forward(train_data);
		tp_count += model.get_accuracy(train_target);

		// back propagation
		model.backward(train_target);

		// update parameters with learning rate decay
		learning_rate *= 1.f / (1.f + lr_decay * step);
		model.update(learning_rate);

		// fetch next data
		step = train_data_loader.next();

		// nvtx profiling end
		nvtxRangePop();

		// calculate softmax loss
		if (step % monitoring_step == 0)
		{
			float loss = model.loss(train_target);
			float accuracy = 100.f * tp_count / monitoring_step / batch_size_train;

			std::cout << "Step: " << std::right << std::setw(4) << step << ", loss: " << std::left << std::setw(5) << std::fixed << std::setprecision(3) << loss << ", accuracy: " << accuracy << "%\n";

			tp_count = 0;
		}
	}

	// save trained parameters
	if (file_save)
	{
		model.write_file(parameter_location);
	}

	// Phase 2: Inferencing
	// Step 1: Load test set
	std::cout << "[INFERENCE]\n";
	MNIST test_data_loader = MNIST("E:/Datasets/mnist");
	test_data_loader.test(batch_size_test);

	// Step 2: Model initialization
	model.test();

	// Step 3: Iterate over the testing loop
	Blob<float>* test_data = test_data_loader.get_data();
	Blob<float>* test_target = test_data_loader.get_target();

	test_data_loader.get_batch();

	tp_count = 0;
	step = 0;
	while (step < num_steps_test)
	{
		// nvtx profiling start
		std::string nvtx_message = std::string("step" + std::to_string(step));
		nvtxRangePushA(nvtx_message.c_str());

		// update shared buffer contents
		test_data->to(cuda);
		test_target->to(cuda);

		// forward
		model.forward(test_data);
		tp_count += model.get_accuracy(test_target);

		// fetch next data
		step = test_data_loader.next();

		// nvtx profiling end
		nvtxRangePop();
	}

	// stop Nsight system profiling
	cudaProfilerStop();

	// Step 4: Calculate loss and accuracy metrics
	float loss = model.loss(test_target);
	float accuracy = 100.f * tp_count / num_steps_test / batch_size_test;

	std::cout << "Loss: " << std::setw(4) << loss << ", accuracy: " << accuracy << "%\n";

	// End
	std::cout << "Done.\n";

	return 0;
}