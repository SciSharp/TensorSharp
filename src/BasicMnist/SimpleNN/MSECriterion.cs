using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using TensorSharp;
using TensorSharp.Expression;

namespace BasicMnist.SimpleNN
{
    public class MSECriterion : ICriterion
    {
        private NDArray output;
        private NDArray gradInput;


        public MSECriterion(IAllocator allocator, int batchSize, int outputSize)
        {
            this.output = new NDArray(allocator, DType.Float32, 1);
            this.gradInput = new NDArray(allocator, DType.Float32, batchSize, outputSize);
        }

        public NDArray UpdateOutput(NDArray input, NDArray target)
        {
            (input.TVar() - target)
                .Pow(2)
                .MeanAll()
                .Evaluate(output);

            return output;
        }

        public NDArray UpdateGradInput(NDArray input, NDArray target)
        {
            var norm = 2.0f / input.ElementCount();

            ((input.TVar() - target) * norm)
                .Evaluate(gradInput);

            return gradInput;
        }
    }
}
