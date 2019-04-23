using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using TensorSharp;

namespace BasicMnist.SimpleNN
{
    public interface ICriterion
    {
        NDArray UpdateOutput(NDArray input, NDArray target);
        NDArray UpdateGradInput(NDArray input, NDArray target);
    }
}
