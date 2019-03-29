using BasicMnist.SimpleNN;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.CUDA;
using TensorSharp.Expression;

namespace BasicMnist
{
    public enum ModelType { MLP, MLPSoftmax, Cnn }
    public enum AccelMode { Cpu, Cuda, Cudnn }


    class Program
    {

        //##########################################################################
        // Configuration options

        private const string MnistFolder = @"C:\MNIST\";
        private const AccelMode AccMode = AccelMode.Cpu;
        private const ModelType MType = ModelType.MLPSoftmax;
        
        // Set these to non-null to restrict how many samples are loaded for each set
        private static readonly int? TRAINING_SIZE = null;
        private static readonly int? TESTING_SIZE = null;
        
        private const int BatchSize = 50;

        private static readonly SgdConfig sgdConfig = new SgdConfig()
        {
            LearningRate = MType == ModelType.Cnn ? 0.001f : 0.1f,
            Momentum = 0.9f,
        };


        // End of configuraion options
        //##########################################################################


        
        static void Main(string[] args)
        {
            // Init TensorSharp

            IAllocator allocator = null;
            if (AccMode == AccelMode.Cpu)
            {
                allocator = new CpuAllocator();
            }
            else
            {
                var cudaContext = new TSCudaContext();
                cudaContext.Precompile(Console.Write);
                cudaContext.CleanUnusedPTX();
                allocator = new CudaAllocator(cudaContext, 0);
            }

            var random = new SeedSource(42); // set seed to a known value - we do this to make the training repeatable



            // Load data

            if (string.IsNullOrEmpty(MnistFolder)) throw new ApplicationException("MnistFolder should be set to the path containing the MNIST data set");

            Console.WriteLine("loading data sets");
            DataSet trainingSet, testingSet;
            using (new SimpleTimer("data set loading done in {0}ms"))
            {
                MnistDataSetBuilder.BuildDataSets(allocator, MnistFolder, TRAINING_SIZE, TESTING_SIZE, out trainingSet, out testingSet);
            }


            // Construct the model, loss function and optimizer
            
            int numInputs = MnistParser.ImageSize * MnistParser.ImageSize;

            Sequential model;
            ICriterion criterion;
            bool useTargetClasses;

            var useCudnn = AccMode == AccelMode.Cudnn;
            switch(MType)
            {
                case ModelType.MLP: ModelBuilder.BuildMLP(allocator, random, BatchSize, useCudnn, out model, out criterion, out useTargetClasses); break;
                case ModelType.MLPSoftmax: ModelBuilder.BuildMLPSoftmax(allocator, random, BatchSize, useCudnn, out model, out criterion, out useTargetClasses); break;
                case ModelType.Cnn: ModelBuilder.BuildCnn(allocator, random, BatchSize, useCudnn, out model, out criterion, out useTargetClasses); break;

                default: throw new InvalidOperationException("Unrecognized model type " + MType);
            }

            var optim = new SgdOptimizer(sgdConfig);


            // Train the model

            for (int i = 0; i < 50; ++i)
            {
                TrainEpoch(model, criterion, optim, trainingSet, numInputs, useTargetClasses);
                EvaluateModel(model, testingSet, numInputs);
            }
        }


        // Runs a single epoch of training.
        private static void TrainEpoch(Sequential model, ICriterion criterion, SgdOptimizer optim, DataSet trainingSet, int numInputs, bool useTargetClasses)
        {
            using (new SimpleTimer("Training epoch completed in {0}ms"))
            {
                for (int batchStart = 0; batchStart <= trainingSet.inputs.Sizes[0] - BatchSize; batchStart += BatchSize)
                {
                    Console.Write(".");

                    var grad = new GradFunc(parameters =>
                    {
                        using (var mbInputs = trainingSet.inputs.Narrow(0, batchStart, BatchSize))
                        using (var mbTargets = trainingSet.targets.Narrow(0, batchStart, BatchSize))
                        using (var mbTargetClasses = trainingSet.targetValues.Narrow(0, batchStart, BatchSize))
                        {
                            foreach (var gradTensor in model.GetGradParameters())
                            {
                                Ops.Fill(gradTensor, 0);
                            }

                            var modelOutput = model.Forward(mbInputs, ModelMode.Train);
                            var criterionOutput = criterion.UpdateOutput(modelOutput, useTargetClasses ? mbTargetClasses : mbTargets);


                            var criterionGradIn = criterion.UpdateGradInput(modelOutput, useTargetClasses ? mbTargetClasses : mbTargets);
                            model.Backward(mbInputs, criterionGradIn, ModelMode.Train);

                            return new OutputAndGrads() { output = modelOutput, grads = model.GetGradParameters().ToArray() };
                        }

                    });

                    optim.Update(grad, model.GetParameters().ToArray());

                }
            }
            Console.WriteLine();
        }


        // Evaluate the model on the test set. Prints the total number of training samples that are classified correctly
        private static void EvaluateModel(Sequential model, DataSet testingSet, int numInputs)
        {
            float totalCorrect = 0;
            for (int batchStart = 0; batchStart <= testingSet.inputs.Sizes[0] - BatchSize; batchStart += BatchSize)
            {
                using (var mbInputs = testingSet.inputs.Narrow(0, batchStart, BatchSize))
                using (var mbTargetValues = testingSet.targetValues.Narrow(0, batchStart, BatchSize))
                {
                    var modelOutput = model.Forward(mbInputs, ModelMode.Evaluate);

                    totalCorrect += (modelOutput.TVar().Argmax(1) == mbTargetValues)
                        .SumAll()
                        .ToScalar()
                        .Evaluate();
                }
            }

            Console.WriteLine("Test set total correct: " + totalCorrect + " / " + testingSet.inputs.Sizes[0]);
        }
    }
}
