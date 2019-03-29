# TensorSharp

TensorSharp is an open source library providing efficient N-dimensional arrays for .NET. Currently, there are backends for running tensor computations on the CPU, and on CUDA GPU hardware. The library is designed to be efficient, easy to install/deploy, and simple to extend. The implementations of many of the core operators are based on the Torch7 project.

## Installation
 1. Either build the project following the instructions in the *Building* section below, or download the prebuilt binaries from the [Releases](https://github.com/alex-weaver/TensorSharp/releases) page.
 2. To add TensorSharp CPU support to a project, add a reference to build/win_x64/TensorSharp/TensorSharp.dll. Copy all the remaining .dll files from build/win_x64/TensorSharp/ to the project's output directory.
 3. To add CUDA support, add a reference to all the .dll files in build/win_x64/TensorSharp.CUDA/. If you have precompiled the CUDA kernels, then copy the build/cuda_cache folder to the project's output directory.
 4. For cuDNN support, follow all previous steps, then obtain the cuDNN 5 .dll from NVIDIA's Accelerated Computing Developer Program, and place it in the project's output directory.

## Platform Support
 - Currently only Windows x64 is supported
 - Requires .NET Standard 2.0


## Building
Prerequisites:
 1. Visual Studio 2017 Community Edition (or another edition of Visual Studio 2017) must be installed
 2. nuget.exe must exist on your PATH
 3. Powershell must be installed

Building:
 1. Run build_win_x64.bat in the root directory.
 2. *(optional)* CUDA kernels are built on demand with NVRTC and cached as .ptx files in the executable directory. To precompile the kernels so that no compilation is needed at runtime, run *precompile_cuda.bat* in the root directory. This will create a folder build/cuda_cache which contains the precompiled kernels.

To run the unit tests, the architecture for the unit tests must be set to x64. To change the architecture setting, go to the Test -> Test Settings -> Default Processor Architecture menu; otherwise, the Test Explorer will not discover the tests.

## MNIST Example
The BasicMnist project contains a basic implementation of some neural network layers and SGD optimizer. It includes 3 different network architectures for classifying the MNIST digits data set. The configuration options for this sample are at the top of Program.cs

Running this project required the MNIST data set to be placed *uncompressed* into C:\MNIST. The folder where the code will look for the data set is configurable from Program.cs. You can obtain the gzipped data files from http://yann.lecun.com/exdb/mnist/


To use cuDNN with this project, you must obtain the cudnn 5 .dll from NVIDIA and place it in the output directory.

The three implemented network architectures are:
1. A two-layer sigmoid network
2. A two-layer network comprising one sigmoid layer and one softmax layer
3. A convolutional network with two convolutional layers with ReLU activations, two fully connected layers with ReLU activations, a softmax on the output and dropout on the fully connected layers.

## Architecture

### Tensors

Tensor objects in TensorSharp are regarded as views over a particular instance of Storage. A Storage object represents a contiguous block of memory of a given element type and length, either in main memory or on a specific GPU. Storage is an abstract class with an implementation for each supported backend. A Tensor object holds a reference to exactly one Storage object, and describes how the memory should be interpreted as a tensor. The key elements of the Tensor object are:
 * **Sizes** - an array holding the size of each dimension of the tensor
 * **Strides** - an array holding the memory stride of each dimension of the tensor
 * **StorageOffset** - the offset from the start of the Storage that holds the first element of the tensor.

### Memory Management
Storage is a reference counted object - each Tensor that holds a reference to the Storage increments its reference count. Tensors implement IDisposable; when disposed, the Tensor releases the reference it holds to the Storage object.

Storage objects must be constructed using an instance of IAllocator - normally this is done indirectly by the Tensor object on initialization. Using an interface for IAllocator provides a way to write device-agnostic code.

### Expression Layer

Manually managing the allocation and disposal of temporary Tensor objects is tedious for expressions involving several operations. To mitigate this, the TensorSharp.Expression namespace provides the TVar and SVar classes. Both classes represent a lazily-evaluated expression; when the expression is evaluated, tensors holding intermediate results are automatically allocated and freed, making it easy to build large composite expressions. TVar represents an expression where the result is a tensor, and SVar represents an expression where the result is a scalar.

There are two ways of converting an existing Tensor object to a TVar to use in an expression:
1. Use a cast - an implicit cast operator to convert Tensor to TVar is provided.
2. Use the TVar() extension method on Tensor. This requires importing the TensorSharp.Expression namespace

## Quickstart
The following example creates a CPU Tensor from an array, adds 2 to each element and prints the tensor:
```
var allocator = new CpuAllocator();

var tensor = Tensor.FromArray(allocator, new float[] { 1, 2, 3, 4 });
Ops.Add(tensor, tensor, 2);
Console.WriteLine(tensor.Format());
tensor.Dispose();
```
Running the same operation on a CUDA GPU only requires constructing a different allocator:
```
var cudaContext = new TSCudaContext();
cudaContext.Precompile();
cudaContext.CleanUnusedPTX();
var allocator = new CudaAllocator(cudaContext, 0);

var tensor = Tensor.FromArray(allocator, new float[] { 1, 2, 3, 4 });
Ops.Add(tensor, tensor, 2);
Console.WriteLine(tensor.Format());
tensor.Dispose();
```
The calls to Precompile and CleanUnusedPTX ensure that all required CUDA kernels have been compiled so there will not be any additional latency when invoking ops.

