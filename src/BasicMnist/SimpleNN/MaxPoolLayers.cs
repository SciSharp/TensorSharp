using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.CUDA;

namespace BasicMnist.SimpleNN
{
    public abstract class MaxPoolLayer : Layer
    {
        protected readonly ConvolutionDesc2d cd;
        protected readonly bool ceilMode;

        protected readonly NDArray activation, indices, gradInput;



        public MaxPoolLayer(IAllocator allocator, DType elementType, int batchSize, long nInputPlane, long inputWidth, long inputHeight, ConvolutionDesc2d cd, bool ceilMode = true)
        {
            this.cd = cd;
            this.ceilMode = ceilMode;

            var inputSizes = new long[] { batchSize, nInputPlane, inputWidth, inputHeight };
            var outputSizes = CpuMaxPoolingOps.OutputSize(inputSizes, ceilMode, cd);
            this.OutputSizes = outputSizes;

            this.activation = new NDArray(allocator, elementType, outputSizes);
            this.indices = new NDArray(allocator, elementType, outputSizes);
            this.gradInput = new NDArray(allocator, elementType, inputSizes);
        }

        public override NDArray Output { get { return activation; } }
        public override NDArray GradInput { get { return gradInput; } }


        public long[] OutputSizes { get; private set; }


        public override IEnumerable<NDArray> GetParameters()
        {
            return Enumerable.Empty<NDArray>();
        }

        public override IEnumerable<NDArray> GetGradParameters()
        {
            return Enumerable.Empty<NDArray>();
        }

        public override void FlattenParams(NDArray parameters, NDArray gradParameters)
        {
            // no parameters
        }

    }



    public class MaxPoolCpu : MaxPoolLayer
    {
        public MaxPoolCpu(IAllocator allocator, DType elementType, int batchSize, long nInputPlane, long inputWidth, long inputHeight, ConvolutionDesc2d cd, bool ceilMode = true)
            : base(allocator, elementType, batchSize, nInputPlane, inputWidth, inputHeight, cd, ceilMode)
        {

        }

        public override NDArray Forward(NDArray input, ModelMode mode)
        {
            CpuMaxPoolingOps.SpatialMaxPoolingForward(input, activation, indices, cd, ceilMode);
            return activation;
        }

        public override NDArray Backward(NDArray input, NDArray gradOutput, ModelMode mode)
        {
            CpuMaxPoolingOps.SpatialMaxPoolingBackward(input, gradOutput, gradInput, indices, cd, ceilMode);
            return gradInput;
        }
    }

    public class MaxPoolCuda : MaxPoolLayer
    {
        private readonly TensorSharp.CUDA.DeviceCode.SpatialMaxPoolKernels maxPool = new TensorSharp.CUDA.DeviceCode.SpatialMaxPoolKernels();

        public MaxPoolCuda(IAllocator allocator, DType elementType, int batchSize, long nInputPlane, long inputWidth, long inputHeight, ConvolutionDesc2d cd, bool ceilMode = true)
            : base(allocator, elementType, batchSize, nInputPlane, inputWidth, inputHeight, cd, ceilMode)
        {

        }

        public override NDArray Forward(NDArray input, ModelMode mode)
        {
            maxPool.SpatialMaxPoolingForward(input, activation, indices, cd, ceilMode);
            return activation;
        }

        public override NDArray Backward(NDArray input, NDArray gradOutput, ModelMode mode)
        {
            maxPool.SpatialMaxPoolingBackward(input, gradOutput, gradInput, indices, cd, ceilMode);
            return gradInput;
        }
    }

    public class MaxPoolCudnn : MaxPoolLayer
    {
        private readonly DNNPoolingDesc poolingDesc;

        public MaxPoolCudnn(IAllocator allocator, DType elementType, int batchSize, long nInputPlane, long inputWidth, long inputHeight, ConvolutionDesc2d cd, bool ceilMode = true)
            : base(allocator, elementType, batchSize, nInputPlane, inputWidth, inputHeight, cd, ceilMode)
        {
            this.poolingDesc = new DNNPoolingDesc(DNNPoolingMode.Max, cd.kH, cd.kW, cd.padH, cd.padW, cd.dH, cd.dW);
        }

        public override NDArray Forward(NDArray input, ModelMode mode)
        {
            DNN.PoolingForward(poolingDesc, input, activation);
            return activation;
        }

        public override NDArray Backward(NDArray input, NDArray gradOutput, ModelMode mode)
        {
            DNN.PoolingBackward(poolingDesc, input, activation, gradInput, gradOutput);
            return gradInput;
        }
    }
}
