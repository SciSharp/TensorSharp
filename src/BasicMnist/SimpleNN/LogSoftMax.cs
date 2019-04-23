using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using TensorSharp;
using TensorSharp.CUDA;
using TensorSharp.Expression;

namespace BasicMnist.SimpleNN
{
    public class LogSoftMax : Layer
    {
        private readonly NDArray activation, gradInput;

        public LogSoftMax(IAllocator allocator, DType elementType, int nInput, int batchSize)
        {
            this.activation = new NDArray(allocator, elementType, batchSize, nInput);
            this.gradInput = new NDArray(allocator, elementType, batchSize, nInput);
        }


        public override NDArray Output { get { return activation; } }
        public override NDArray GradInput { get { return gradInput; } }
        public override IEnumerable<NDArray> GetGradParameters() { return Enumerable.Empty<NDArray>(); }
        public override IEnumerable<NDArray> GetParameters() { return Enumerable.Empty<NDArray>(); }

        public override void FlattenParams(NDArray parameters, NDArray gradParameters)
        {
            // no parameters
        }

        public override NDArray Forward(NDArray input, ModelMode mode)
        {
            var maxes = input.TVar().Max(1);
            var maxesExp = maxes.Expand(input.Shape);

            var d = (input - maxesExp).Exp().Sum(1).Log();
            var logSum = (d + maxes).Expand(input.Shape);

            (input - logSum)
                .Evaluate(activation);

            return activation;
        }

        public override NDArray Backward(NDArray input, NDArray gradOutput, ModelMode mode)
        {
            var go = gradOutput.TVar();
            var a = activation.TVar().Exp().CMul(go.Sum(1).Expand(activation.Shape));
            (go - a)
                .Evaluate(gradInput);

            return gradInput;
        }
    }

    public class LogSoftMaxDNN : Layer
    {
        private readonly NDArray activation, gradInput;

        public LogSoftMaxDNN(IAllocator allocator, DType elementType, int nInput, int batchSize)
        {
            this.activation = new NDArray(allocator, elementType, batchSize, nInput);
            this.gradInput = new NDArray(allocator, elementType, batchSize, nInput);
        }


        public override NDArray Output { get { return activation; } }
        public override NDArray GradInput { get { return gradInput; } }
        public override IEnumerable<NDArray> GetGradParameters() { return Enumerable.Empty<NDArray>(); }
        public override IEnumerable<NDArray> GetParameters() { return Enumerable.Empty<NDArray>(); }

        public override void FlattenParams(NDArray parameters, NDArray gradParameters)
        {
            // no parameters
        }

        private NDArray As4d(NDArray value)
        {
            return value.View(value.Shape[0], value.Shape[1], 1, 1);
        }

        public override NDArray Forward(NDArray input, ModelMode mode)
        {
            using (var input4d = As4d(input))
            using (var activation4d = As4d(activation))
            {
                DNN.SoftmaxForward(DNNSoftmaxAlgorithm.Log, DNNSoftmaxMode.Instance, input4d, activation4d);
            }

            return activation;
        }

        public override NDArray Backward(NDArray input, NDArray gradOutput, ModelMode mode)
        {
            using (var activation4d = As4d(activation))
            using (var gradInput4d = As4d(gradInput))
            using (var gradOutput4d = As4d(gradOutput))
            {
                DNN.SoftmaxBackward(DNNSoftmaxAlgorithm.Log, DNNSoftmaxMode.Instance, activation4d, gradInput4d, gradOutput4d);
            }

            return gradInput;
        }

    }
}
