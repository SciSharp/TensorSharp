using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorSharp;
using TensorSharp.Cpu;

namespace TensorSharpTests
{
    [TestClass]
    public class CopyOpTests
    {
        [TestMethod]
        public void SetGetFloat() { RunSetGet(DType.Float32); }
        [TestMethod]
        public void SetGetDouble() { RunSetGet(DType.Float64); }
        [TestMethod]
        public void SetGetInt() { RunSetGet(DType.Int32); }
        [TestMethod]
        public void SetGetByte() { RunSetGet(DType.UInt8); }

        private void RunSetGet(DType type)
        {
            var allocator = new CpuAllocator();
            var a = new NDArray(allocator, DType.Float32, 1);

            var value = 123.0f;
            a.SetElementAsFloat(value, 0);

            Assert.AreEqual(value, a.GetElementAsFloat(0));
        }
         

        [TestMethod]
        public void CopyFloatToFloat()
        {
            RunCopy(new float[] { 12, 30, 2, 255 }, DType.Float32);
        }

        [TestMethod]
        public void CopyFloatToDouble()
        {
            RunCopy(new float[] { 12, 30, 2, 255 }, DType.Float64);
        }

        [TestMethod]
        public void CopyByteToFloat()
        {
            RunCopy(new byte[] { 12, 30, 2, 255 }, DType.Float32);
        }

        [TestMethod]
        public void CopyFloatToByte()
        {
            RunCopy(new float[] { 12, 30, 2, 255 }, DType.UInt8);
        }

        private void RunCopy(Array srcData, DType destType)
        {
            var allocator = new CpuAllocator();
            var a = NDArray.FromArray(allocator, srcData);
            var b = new NDArray(allocator, destType, a.Shape);

            Ops.Copy(b, a);

            for (int i = 0; i < srcData.Length; ++i)
            {
                Assert.AreEqual(Convert.ToSingle(srcData.GetValue(i)), b.GetElementAsFloat(i));
            }
        }
    }
}
