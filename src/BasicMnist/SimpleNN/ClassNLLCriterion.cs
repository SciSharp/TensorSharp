using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using TensorSharp;
using TensorSharp.Expression;

namespace BasicMnist.SimpleNN
{
    public class ClassNLLCriterion : ICriterion
    {
        private readonly IAllocator allocator;
        private NDArray output;
        private NDArray gradInput;


        public ClassNLLCriterion(IAllocator allocator, int batchSize, int nClasses)
        {
            this.allocator = allocator;

            this.output = new NDArray(allocator, DType.Float32, 1);
            this.gradInput = new NDArray(allocator, DType.Float32, batchSize, nClasses);
        }

        public NDArray UpdateOutput(NDArray input, NDArray target)
        {
            var indices = target.TVar().View(target.Shape[0], 1);

            var loss = input.TVar()
                .Gather(1, indices)
                .SumAll()
                 * (-1.0f / target.Shape[0]);

            loss.Evaluate(output);

            return output;
        }

        public NDArray UpdateGradInput(NDArray input, NDArray target)
        {
            var norm = -1.0f / input.Shape[0];

            Variable.Fill(0, allocator, DType.Float32, gradInput.Shape)
                .Evaluate(gradInput);
            
            var indices = target.TVar().View(target.Shape[0], 1);
            
            gradInput.TVar()
                .ScatterFill(norm, 1, indices)
                .Evaluate(gradInput);
            

            return gradInput;
        }
    }
}
