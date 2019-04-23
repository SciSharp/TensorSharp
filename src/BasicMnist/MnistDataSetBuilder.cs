using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using TensorSharp;
using TensorSharp.Expression;

namespace BasicMnist
{
    public class DataSet
    {
        public NDArray inputs;
        public NDArray targets;
        public NDArray targetValues;
    }

    public static class MnistDataSetBuilder
    {
        private const string MnistTrainImages = "train-images.idx3-ubyte";
        private const string MnistTrainLabels = "train-labels.idx1-ubyte";
        private const string MnistTestImages = "t10k-images.idx3-ubyte";
        private const string MnistTestLabels = "t10k-labels.idx1-ubyte";

        public static void BuildDataSets(IAllocator allocator, string baseFolder, int? maxTrainImages, int? maxTestImages, out DataSet trainingData, out DataSet testingData)
        {
            var trainingImages = MnistParser.Parse(
                Path.Combine(baseFolder, MnistTrainImages),
                Path.Combine(baseFolder, MnistTrainLabels),
                maxTrainImages);

            var testImages = MnistParser.Parse(
                Path.Combine(baseFolder, MnistTestImages),
                Path.Combine(baseFolder, MnistTestLabels),
                maxTestImages);

            trainingData = BuildSet(allocator, trainingImages);
            testingData = BuildSet(allocator, testImages);
        }

        public static DataSet BuildSet(IAllocator allocator, DigitImage[] images)
        {
            var inputs = new NDArray(allocator, DType.Float32, images.Length, MnistParser.ImageSize, MnistParser.ImageSize);
            var outputs = new NDArray(allocator, DType.Float32, images.Length, MnistParser.LabelCount);

            var cpuAllocator = new TensorSharp.Cpu.CpuAllocator();
            
            for (int i = 0; i < images.Length; ++i)
            {
                var target = inputs.TVar().Select(0, i);

                Variable.FromArray(images[i].pixels, cpuAllocator)
                    .AsType(DType.Float32)
                    .ToDevice(allocator)
                    .Evaluate(target);

                target.Div(255)
                    .Evaluate(target);
            }
            

            Ops.FillOneHot(outputs, MnistParser.LabelCount, images.Select(x => (int)x.label).ToArray());
            var targetValues = NDArray.FromArray(allocator, images.Select(x => (float)x.label).ToArray());
            
            return new DataSet() { inputs = inputs, targets = outputs, targetValues = targetValues };
        }
    }
}
