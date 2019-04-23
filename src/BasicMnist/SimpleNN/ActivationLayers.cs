using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using TensorSharp;
using TensorSharp.Expression;

namespace BasicMnist.SimpleNN
{
    public abstract class ActivationLayer : Layer
    {
        protected readonly NDArray activation, gradInput;

        public ActivationLayer(IAllocator allocator, DType elementType, params long[] shape)
        {
            this.activation = new NDArray(allocator, elementType, shape);
            this.gradInput = new NDArray(allocator, elementType, shape);
        }

        public override NDArray Output { get { return activation; } }
        public override NDArray GradInput { get { return gradInput; } }

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
            // no parameters in activation layers
        }
    }

    public class SigmoidLayer : ActivationLayer
    {
        public SigmoidLayer(IAllocator allocator, DType elementType, params long[] shape)
            : base(allocator, elementType, shape)
        {
        }

        public override NDArray Forward(NDArray input, ModelMode mode)
        {
            Ops.Sigmoid(activation, input);
            return activation;
        }

        public override NDArray Backward(NDArray input, NDArray gradOutput, ModelMode mode)
        {
            UpdateGradInput(gradOutput, activation);
            return gradInput;
        }

        private void UpdateGradInput(NDArray gradOutput, NDArray output)
        {
            // Computes  gradInput = gradOutput .* (1 - output) .* output

            gradOutput.TVar()
                .CMul(1 - output.TVar())
                .CMul(output)
                .Evaluate(gradInput);

       }

    }

    /// <summary>
    /// Output element x' -> x if x > threshold; val otherwise
    /// </summary>
    public class ThresholdLayer : ActivationLayer
    {
        private readonly float threshold, val;

        public ThresholdLayer(IAllocator allocator, DType elementType, long[] shape, float threshold, float val)
            : base(allocator, elementType, shape)
        {
            this.threshold = threshold;
            this.val = val;
        }

        public override NDArray Forward(NDArray input, ModelMode mode)
        {
            var keepElements = input.TVar() > threshold;
            (input.TVar().CMul(keepElements) + (1 - keepElements) * val)
                .Evaluate(activation);
            
            return activation;
        }

        public override NDArray Backward(NDArray input, NDArray gradOutput, ModelMode mode)
        {
            UpdateGradInput(input, gradOutput);
            return gradInput;
        }

        private void UpdateGradInput(NDArray input, NDArray gradOutput)
        {
            // Retains gradients only where input x > threshold

            gradOutput.TVar().CMul(input.TVar() > threshold)
                .Evaluate(gradInput);
        }
    }

    public class ReLULayer : ThresholdLayer
    {
        public ReLULayer(IAllocator allocator, DType elementType, params long[] shape)
            : base(allocator, elementType, shape, 0, 0)
        {
        }
    }
}
