using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using TensorSharp;

namespace BasicMnist.SimpleNN
{
    public class Sequential : Layer
    {
        private readonly List<Layer> layers = new List<Layer>();

        private NDArray lastOutput, lastGradInput;

        public Sequential()
        {
        }

        public override NDArray Output { get { return lastOutput; } }
        public override NDArray GradInput { get { return lastGradInput; } }


        public void Add(Layer layer)
        {
            this.layers.Add(layer);
        }

        public override IEnumerable<NDArray> GetParameters()
        {
            foreach (var layer in layers)
            {
                foreach (var p in layer.GetParameters())
                {
                    yield return p;
                }
            }
        }

        public override IEnumerable<NDArray> GetGradParameters()
        {
            foreach (var layer in layers)
            {
                foreach (var p in layer.GetGradParameters())
                {
                    yield return p;
                }
            }
        }

        public override void FlattenParams(NDArray parameters, NDArray gradParameters)
        {
            long offset = 0;
            for (int i = 0; i < layers.Count; i++)
            {
                var layer = layers[i];

                using (var paramSlice = parameters.Narrow(0, offset, layer.GetParameterCount()))
                using (var gradParamSlice = gradParameters.Narrow(0, offset, layer.GetParameterCount()))
                {
                    layer.FlattenParams(paramSlice, gradParamSlice);
                }

                offset += layer.GetParameterCount();
            }
        }

        public override NDArray Forward(NDArray input, ModelMode mode)
        {
            NDArray curOutput = input;
            foreach (var layer in layers)
            {
                curOutput = layer.Forward(curOutput, mode);
            }

            lastOutput = curOutput;
            return curOutput;
        }

        public override NDArray Backward(NDArray input, NDArray gradOutput, ModelMode mode)
        {
            var curGradOutput = gradOutput;

            for (int i = layers.Count - 1; i > 0; --i)
            {
                var layer = layers[i];
                var prevLayer = layers[i - 1];

                curGradOutput = layer.Backward(prevLayer.Output, curGradOutput, mode);
            }

            curGradOutput = layers[0].Backward(input, curGradOutput, mode);

            lastGradInput = curGradOutput;
            return curGradOutput;
        }

    }
}
