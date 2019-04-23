using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using TensorSharp;
using TensorSharp.Expression;

namespace BasicMnist.SimpleNN
{
    public class DropoutLayer : Layer
    {
        private readonly SeedSource seedSource;
        private readonly IAllocator allocator;
        private readonly DType elementType;

        private readonly NDArray activation, gradInput, noise;
        private readonly float pRemove;


        public DropoutLayer(IAllocator allocator, SeedSource seedSource, DType elementType, float pRemove, params long[] shape)
        {
            this.seedSource = seedSource;
            this.allocator = allocator;
            this.elementType = elementType;

            this.pRemove = pRemove;

            this.activation = new NDArray(allocator, elementType, shape);
            this.gradInput = new NDArray(allocator, elementType, shape);
            this.noise = new NDArray(allocator, elementType, shape);
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
            // no parameters
        }

        public override NDArray Forward(NDArray input, ModelMode mode)
        {
            Ops.Copy(activation, input);

            if (mode == ModelMode.Train)
            {
                var p = 1 - pRemove;

                Variable.RandomBernoulli(seedSource, p, allocator, elementType, noise.Shape)
                    .Div(p)
                    .Evaluate(noise);

                activation.TVar()
                    .CMul(noise)
                    .Evaluate(activation);
            }

            return activation;
        }

        public override NDArray Backward(NDArray input, NDArray gradOutput, ModelMode mode)
        {   
            UpdateGradInput(gradOutput, activation, mode == ModelMode.Train);
            return gradInput;
        }

        private void UpdateGradInput(NDArray gradOutput, NDArray output, bool train)
        {
            Ops.Copy(gradInput, gradOutput);

            if (train)
            {
                Ops.Mul(gradInput, gradInput, noise);
            }
        }
    }
}
