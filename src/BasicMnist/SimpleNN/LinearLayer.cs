using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using TensorSharp;
using TensorSharp.Expression;

namespace BasicMnist.SimpleNN
{
    public class LinearLayer : Layer
    {
        private NDArray weights, bias, activation, gradInput;
        private NDArray gradWeights, gradBias;

        private readonly int batchSize, nOutput;

        public LinearLayer(IAllocator allocator, SeedSource seedSource, DType elementType, int nInput, int nOutput, int batchSize)
        {
            this.batchSize = batchSize;
            this.nOutput = nOutput;

            this.weights = new NDArray(allocator, elementType, nInput, nOutput);
            this.bias = new NDArray(allocator, elementType, 1, nOutput);

            this.activation = new NDArray(allocator, elementType, batchSize, nOutput);

            this.gradInput = new NDArray(allocator, elementType, batchSize, nInput);
            this.gradWeights = new NDArray(allocator, elementType, nInput, nOutput);
            this.gradBias = new NDArray(allocator, elementType, 1, nOutput);

            InitWeightsLinear(seedSource, weights, bias);
        }

        public override NDArray Output { get { return activation; } }
        public override NDArray GradInput { get { return gradInput; } }

        public override IEnumerable<NDArray> GetParameters()
        {
            yield return weights;
            yield return bias;
        }

        public override IEnumerable<NDArray> GetGradParameters()
        {
            yield return gradWeights;
            yield return gradBias;
        }

        public override void FlattenParams(NDArray parameters, NDArray gradParameters)
        {
            var weightSize = weights.ElementCount();
            var biasSize = bias.ElementCount();

            weights.TVar().View(weightSize)
                .Evaluate(parameters.TVar().Narrow(0, 0, weightSize));

            bias.TVar().View(biasSize)
                .Evaluate(parameters.TVar().Narrow(0, weightSize, biasSize));

            gradWeights.TVar().View(weightSize)
                .Evaluate(gradParameters.TVar().Narrow(0, 0, weightSize));

            gradBias.TVar().View(biasSize)
                .Evaluate(gradParameters.TVar().Narrow(0, weightSize, biasSize));
        }

        public override NDArray Forward(NDArray input, ModelMode mode)
        {
            // activation = [bias] + input * weights
            // where [bias] means broadcast the bias vector
            bias.TVar().Expand(batchSize, nOutput)
                .Addmm(1, 1, input, weights)
                .Evaluate(activation);

            return activation;
        }

        public override NDArray Backward(NDArray input, NDArray gradOutput, ModelMode mode)
        {
            UpdateGradInput(gradOutput);
            AccWeightGrads(input, gradOutput);
            return gradInput;
        }

        private void AccWeightGrads(NDArray input, NDArray gradOutput)
        {
            gradWeights.TVar().Addmm(1, 1, input.TVar().Transpose(), gradOutput)
                .Evaluate(gradWeights);

            (gradBias + gradOutput.TVar().Sum(0))
                .Evaluate(gradBias);
        }

        private void UpdateGradInput(NDArray gradOutput)
        {
            gradOutput.TVar().Dot(weights.TVar().Transpose())
                .Evaluate(gradInput);
        }

        private void InitWeightsLinear(SeedSource seedSource, NDArray weights, NDArray bias)
        {
            var stdv = 1.0f / (float)Math.Sqrt(weights.Shape[1]);
            Ops.RandomUniform(weights, seedSource, -stdv, stdv);
            Ops.RandomUniform(bias, seedSource, -stdv, stdv);
        }
    }
}
