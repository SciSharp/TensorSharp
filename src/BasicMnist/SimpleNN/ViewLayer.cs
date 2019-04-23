using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using TensorSharp;

namespace BasicMnist.SimpleNN
{
    public class ViewLayer : Layer
    {
        private readonly long[] resultSize;

        private long[] lastInputSize;
        private NDArray activation, gradInput;

        public ViewLayer(params long[] resultSize)
        {
            this.resultSize = resultSize;
        }

        protected ViewLayer()
        {
        }

        public override NDArray GradInput { get { return gradInput; } }
        public override NDArray Output { get { return activation; } }

        public override NDArray Backward(NDArray input, NDArray gradOutput, ModelMode mode)
        {
            if (gradInput != null)
                gradInput.Dispose();

            gradInput = gradOutput.View(lastInputSize);
            return gradInput;
        }

        public override NDArray Forward(NDArray input, ModelMode mode)
        {
            if (activation != null)
                activation.Dispose();
            activation = input.View(resultSize);
            lastInputSize = input.Shape;
            return activation;
        }

        public override IEnumerable<NDArray> GetGradParameters()
        {
            return Enumerable.Empty<NDArray>();
        }

        public override IEnumerable<NDArray> GetParameters()
        {
            return Enumerable.Empty<NDArray>();
        }

        public override void FlattenParams(NDArray parameters, NDArray gradParameters)
        {
            // no parameters
        }
    }
}
