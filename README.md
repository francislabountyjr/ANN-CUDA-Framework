# ANN-CUDA-Framework
Basic ANN Framework utilizing cudnn and the CUDA programming language. Sources: "Learn CUDA programming" from Jaegeun Han and Bharatkumar Sharma.

Instructions: 
   1. Clone the repo and open the visual studio project. 
   2. Make sure your desired cuda and cudnn versions are linked in VC++ Directories > Library Directories.
   3. Download MNIST from "http://yann.lecun.com/exdb/mnist/" or look at the mnist.cu and mnist.cuh source files to write your own data loader.
   4. Edit train.cu to include your desired network architecture and parameters.
   5. Compile and run the application.

*Model will be a sequential model. Look at train.cu for an example of how to set up the network architecture.
