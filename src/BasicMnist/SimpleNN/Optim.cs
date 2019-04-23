using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using TensorSharp;

namespace BasicMnist.SimpleNN
{
    public struct OutputAndGrads
    {
        public NDArray output;
        public NDArray[] grads;
    }

    public delegate OutputAndGrads GradFunc(NDArray[] parameters);

    public struct SgdConfig
    {
        public float LearningRate;
        public float Momentum;
    }

    public class SgdOptimizer
    {
        private readonly SgdConfig config;

        private NDArray[] gradAcc;


        public SgdOptimizer(SgdConfig config)
        {
            this.config = config;
        }

        public void Reset()
        {
            gradAcc = null;
        }

        // Modifies parameters in place
        // returns model output
        public NDArray Update(GradFunc grad, NDArray[] parameters)
        {
            var outputAndGrads = grad(parameters);
            NDArray output = outputAndGrads.output;
            NDArray[] gradients = outputAndGrads.grads;
            
                if (gradAcc == null)
                {
                    gradAcc = gradients.Select(x =>
                    {
                        var result = new NDArray(x.Allocator, x.ElementType, x.Shape);
                        Ops.Fill(result, 0);
                        return result;
                    }).ToArray();
                }

                // gradAcc = gradAcc * momentum - learningRate * gradients
                for (int i = 0; i < gradients.Length; ++i)
                {
                    Ops.Mul(gradAcc[i], gradAcc[i], config.Momentum);
                    using (var temp = Ops.Mul(null, gradients[i], -config.LearningRate))
                    {
                        Ops.Add(gradAcc[i], gradAcc[i], temp);
                    }
                }
            

            for (int i = 0; i < parameters.Length; ++i)
            {
                Ops.Add(parameters[i], parameters[i], gradAcc[i]);
            }

            return output;
        }
    }
}
