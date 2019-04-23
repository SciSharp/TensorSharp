using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using TensorSharp;
using TensorSharp.Expression;

namespace BasicMnist.SimpleNN
{
    public enum ModelMode
    {
        Train,
        Evaluate,
    }

    public abstract class Layer
    {
        public abstract IEnumerable<NDArray> GetParameters();
        public abstract IEnumerable<NDArray> GetGradParameters();

        public long GetParameterCount()
        {
            return GetParameters().Aggregate(0L, (acc, item) => acc + item.ElementCount());
        }

        public abstract void FlattenParams(NDArray parameters, NDArray gradParameters);


        public abstract NDArray Forward(NDArray input, ModelMode mode);
        public abstract NDArray Backward(NDArray input, NDArray gradOutput, ModelMode mode);

        public abstract NDArray Output { get; }
        public abstract NDArray GradInput { get; }
    }
}
